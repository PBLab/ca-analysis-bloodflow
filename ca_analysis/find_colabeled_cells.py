import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import pathlib
import attr
import enum
from attr.validators import instance_of
import tifffile
import scipy.ndimage
import scipy.spatial.distance
import scipy.stats
import warnings
import skimage.draw, skimage.measure


class TiffChannels(enum.Enum):
    ONE = 0
    TWO = 1
    THREE = 2
    FOUR = 3


@attr.s
class ColabeledCells:
    """ 
    Analyze a TIF stack with two channels, one of them 
    shows functional activity and one is only morphological. The code
    finds the co-labeled cells and returns their indices.
    """
    tif = attr.ib(validator=instance_of(pathlib.Path))
    result_file = attr.ib(validator=instance_of(pathlib.Path))
    activity_ch = attr.ib(validator=instance_of(TiffChannels))
    morph_ch = attr.ib(validator=instance_of(TiffChannels))
    verbose = attr.ib(default=False, validator=instance_of(bool))
    cell_radius = attr.ib(default=12, validator=instance_of(int))
    colabeled_idx = attr.ib(init=False)
    unlabeled_idx = attr.ib(init=False)
    raw_data = attr.ib(init=False)
    num_of_channels = attr.ib(init=False)
    act_data = attr.ib(init=False)
    morph_data = attr.ib(init=False)
    act_img = attr.ib(init=False)
    morph_img = attr.ib(init=False)
    struct_element = attr.ib(init=False)

    def __attrs_post_init__(self):
        assert self.activity_ch != self.morph_ch
        # Divide movie into its channels (supports 2 currently)
        raw_data = tifffile.imread(str(self.tif))
        try:
            self.num_of_channels = len(tifffile.TiffFile(str(self.tif)).scanimage_metadata\
                ['FrameData']['SI.hChannels.channelsActive'])
        except TypeError:
            warnings.warn('Not a ScanImage stack.')
            self.num_of_channels = 1
        self.act_data = raw_data[self.activity_ch.value::self.num_of_channels]
        self.morph_data = raw_data[self.morph_ch.value::self.num_of_channels]
        self.act_img = self.act_data.sum(axis=0)
        self.morph_img = self.morph_data.sum(axis=0)
    
    def find_colabeled(self):
        """ Main method of class. Finds co-labeled cells. """
        if self.verbose:
            self._show_images()
        self.struct_element = self._create_mask(self.cell_radius)
        region_props = self._find_cells(self.morph_img, self.struct_element)
        dist_mat, func_idx, morph_idx = self._filter_regions(region_props)
        min_distances = self._find_unique_pairs(dist_mat, func_idx, morph_idx)
        if self.verbose:
            self._show_colabeled_cells(min_distances)
        
    def _show_images(self):
        """ Show the summed images of each channel of data """
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(self.act_img, cmap='gray')
        ax[0].set_title('Active')
        ax[1].imshow(self.morph_img, cmap='gray')
        ax[1].set_title('Morph')

    def _create_mask(self, r):
        """ Create a round structuring element for scipy.ndimage.label """
        mask = np.zeros((r * 2, r * 2), dtype=np.uint8)
        rr, cc = skimage.draw.circle(r - 1, r - 1, r)
        mask[rr, cc] = 1
        return mask

    def _find_cells(self, img, mask):
        """ Detects cell-like shapes in an xy image """
        assert len(img.shape) == 2
        quantile_val = np.percentile(img, 99.2)
        binary_img = np.zeros_like(img)
        binary_img[img > quantile_val] = 1
        label, num_features = scipy.ndimage.label(binary_img)
        if self.verbose:
            fig, ax = plt.subplots()
            ax.set_title(f'Cells (found {num_features} components)')
            ax.imshow(label)

        return skimage.measure.regionprops(label)

    def _filter_regions(self, regions):
        """ Filter each region in the labeled image by its area and location.
        Returns indices of the components that are close to each other. """

        # Start by reading the center of mass coordinates of CaImAn components
        all_crd = np.load(self.result_file)['crd']
        centroids_functional = np.array([data['CoM'] for data in all_crd])
        assert centroids_functional.shape[1] == 2  # two columns, x and y
        large_regions = [region for region in regions if region.area > self.cell_radius ** 2]
        centroids_morph = np.array([region.centroid for region in large_regions])
        assert centroids_morph.shape[1] == 2

        # Calculate pair-wise distance of all cells, and find the closest ones
        dist = scipy.spatial.distance.cdist(centroids_functional, centroids_morph)
        close_functional_idx, close_morph_idx = np.where(dist < 2 * self.cell_radius)
        if self.verbose:
            print("The distance filter reduced the number of detected morph cells"
                  f" from {len(large_regions)} to {len(np.unique(close_morph_idx))}.")
        return dist, close_functional_idx, close_morph_idx
    
    def _find_unique_pairs(self, dist, func_idx, morph_idx):
        """
        Finds and returns only the unique pairs of morphological and
        functional cells. The matrix it returns has the functional indices in column 0,
        morphological indices in column 1, and the paired distance in column 2.
        """        
        dist_mat = np.full((min(len(np.unique(func_idx)), len(np.unique(morph_idx))), 3), np.nan)
        dist_idx = 0
        for func_i, morph_i in zip(func_idx, morph_idx):
            if self.verbose:
                print(f"dist_idx is {dist_idx}")
            if (func_i not in dist_mat[:, 0]) and (morph_i not in dist_mat[:, 1]):
                dist_mat[dist_idx, :] = (func_i, morph_i, dist[func_i, morph_i])
                dist_idx += 1
                continue
            if func_i in dist_mat[:, 0]:
                dupe_idx = np.where(dist_mat[:, 0] == func_i)[0][0]
                new_dist = dist[func_i, morph_i]
                if new_dist < dist_mat[dupe_idx, 2]:
                    dist_mat[dupe_idx, :] = (func_i, morph_i, new_dist)
                continue
            
            if morph_i in dist_mat[:, 1]:
                dupe_idx = np.where(dist_mat[:, 1] == morph_i)[0][0]
                new_dist = dist[func_i, morph_i]
                if new_dist < dist_mat[dupe_idx, 2]:
                    dist_mat[dupe_idx, :] = (func_i, morph_i, new_dist)

        return dist_mat


if __name__ == '__main__':
    tif = pathlib.Path.home() / pathlib.Path(r'data/Amos/occluder/4th_July18_VIP_Td_SynGCaMP_Occluder/fov1_mag_2p5_256PX_58p28HZ_vasc_occ_00001.tif')
    result = pathlib.Path.home() / pathlib.Path(r'data/Amos/occluder/4th_July18_VIP_Td_SynGCaMP_Occluder/fov1_mag_2p5_256PX_58p28HZ_vasc_occ_00001_CHANNEL_1_results.npz')
    c = ColabeledCells(tif=tif, result_file=result,
                       activity_ch=TiffChannels.ONE, morph_ch=TiffChannels.TWO,
                       verbose=True, cell_radius=4)
    c.find_colabeled()
    plt.show(block=True)
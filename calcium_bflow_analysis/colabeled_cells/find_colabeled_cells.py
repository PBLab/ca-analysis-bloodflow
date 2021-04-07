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
    cell_radius = attr.ib(default=12, validator=instance_of(int))
    verbose = attr.ib(default=False, validator=instance_of(bool))
    colabeled_idx = attr.ib(init=False)
    unlabeled_idx = attr.ib(init=False)
    num_of_channels = attr.ib(init=False)
    act_data = attr.ib(init=False)
    morph_data = attr.ib(init=False)
    act_img = attr.ib(init=False)
    morph_img = attr.ib(init=False)
    struct_element = attr.ib(init=False)

    def __attrs_post_init__(self):
        assert self.activity_ch != self.morph_ch
        # Divide movie into its channels (supports 2 currently)
        try:
            self.num_of_channels = len(tifffile.TiffFile(str(self.tif)).scanimage_metadata\
                ['FrameData']['SI.hChannels.channelsActive'])
        except TypeError:
            warnings.warn('Not a ScanImage stack.')
            self.num_of_channels = 1

        with tifffile.TiffFile(str(self.tif), movie=True) as f:
            if self.verbose:
                raw_data = f.asarray()
                self.morph_data = raw_data[self.morph_ch.value::self.num_of_channels]
                self.act_data = raw_data[self.activity_ch.value::self.num_of_channels]
                self.act_img = self.act_data.sum(axis=0)
            else:
                sl = slice(self.morph_ch.value, None, self.num_of_channels)
                self.morph_data = f.asarray(sl)
        self.morph_img = self.morph_data.sum(axis=0)
        
    def find_colabeled(self):
        """ Main method of class. Finds co-labeled cells. Returns the number
        of cells found. """
        if self.verbose:
            self._show_images()
        self.struct_element = self._create_mask(self.cell_radius)
        region_props = self._find_cells(self.morph_img, self.struct_element)
        dist_mat, func_idx, morph_idx = self._filter_regions(region_props)
        min_distances = self._find_unique_pairs(dist_mat, func_idx, morph_idx)
        if self.verbose:
            self._show_colabeled_cells(min_distances)
        self._serialize_colabeled(min_distances)
        return min_distances.shape[0]

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
        quantile_val = np.percentile(img, 98)
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
        res_file = np.load(self.result_file)
        all_crd = res_file['crd']
        if 'params' not in res_file:  # newer .npz files are already "filtered" and so this line isn't needed
            all_crd = all_crd[res_file['accepted_list']]  # filters bad components
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

    def _show_colabeled_cells(self, min_distances):
        """ Shows a plot of the correlation image with the colabeled cells """
        result_data = np.load(self.result_file)
        crds = result_data['crd']
        colabeled_cells = [crds[int(idx)]['CoM'][::-1] for idx in min_distances[:, 0]]
        circles_0 = [plt.Circle(com, self.cell_radius, alpha=0.3, color='green') for com in colabeled_cells]
        circles_1 = [plt.Circle(com, self.cell_radius, alpha=0.3, color='green') for com in colabeled_cells]
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(self.act_img, cmap='gray')
        [ax[0].add_artist(circle) for circle in circles_0]
        ax[1].imshow(self.morph_img, cmap='gray')
        [ax[1].add_artist(circle) for circle in circles_1]
        fig.savefig(str(self.tif)[:-4] + '.pdf', transparent=True)
    
    def _serialize_colabeled(self, dist):
        """ Write the cell indices to disk """
        fname = pathlib.Path(str(self.result_file)[:-11] + "colabeled_idx.npy")
        try:
            np.save(fname, dist[:, 0].astype(np.uint32))
        except PermissionError:
            warnings.warn(f"Permission error for folder {fname.parent}")


def batch_colabeled(foldername: pathlib.Path, glob='*results.npz', verbose=False):
    """ Batch process all stacks in folder to find and write to disk
    the indices of the colabeled cells """
    result_files = foldername.rglob(glob)
    for file in result_files:
        print(f"Loading {file} and its matching TIF...")
        name_without_channel = str(file.name)[:-22] + '.tif'
        try:
            matching_tif = next(file.parent.glob(name_without_channel))
        except StopIteration:
            continue
        else:
            cur_pair = (file, matching_tif)
            colabeled = ColabeledCells(tif=matching_tif, result_file=file,
                                       activity_ch=TiffChannels.ONE,
                                       morph_ch=TiffChannels.TWO,
                                       cell_radius=5, verbose=verbose).find_colabeled()
            print(f"File {file} contained {colabeled} colabeled cells.")


if __name__ == '__main__':
    plt.show(block=True)
    # folder = pathlib.Path.home() / pathlib.Path(r'data/David/Vascular occluder_ALL/vip_td_gcamp_vasc_occ_anaesthetise')
    folder = pathlib.Path('/data/David/Vascular occluder_ALL/vip_td_gcamp_270818_muscle_only/')
    assert folder.exists()
    glob = r'f*60Hz*results.npz'
    batch_colabeled(folder, glob=glob, verbose=True)

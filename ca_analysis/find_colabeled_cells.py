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
        func_idx, morph_idx = self._filter_regions(region_props)
        unique_func_idx, unique_morph_idx = self._find_unique_pairs(func_idx, morph_idx)

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
        """ Finds cell-like shapes in an xy image """
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
        dist = scipy.spatial.distance.cdist(centroids_functional, centroids_morph)
        close_functional, close_morph = np.where(dist < 2 * self.cell_radius)
        if self.verbose:
            print("The distance filter reduced the number of detected morph cells"
                  f" from {len(large_regions)} to {len(close_morph)}.")
        return close_functional, close_morph
    
    def _find_unique_pairs(self, func_idx, morph_idx):
        """ Finds and returns only the unique pairs of morphological and
        functional cells """
        unique_func, unique_morph = [], []
        for cur_func, cur_morph in zip(func_idx, morph_idx):
            if (cur_func not in unique_func) and (cur_morph not in unique_morph):
                unique_func.append(cur_func)
                unique_morph.append(cur_morph)
                continue
            try:
                idx = unique_func.index(cur_func)

            







        
if __name__ == '__main__':
    tif = pathlib.Path.home() / pathlib.Path(r'data/Amos/occluder/4th_July18_VIP_Td_SynGCaMP_Occluder/fov1_mag_2p5_256PX_58p28HZ_vasc_occ_00001.tif')
    result = pathlib.Path.home() / pathlib.Path(r'data/Amos/occluder/4th_July18_VIP_Td_SynGCaMP_Occluder/fov1_mag_2p5_256PX_58p28HZ_vasc_occ_00001_CHANNEL_1_results.npz')
    c = ColabeledCells(tif=tif, result_file=result,
                       activity_ch=TiffChannels.ONE, morph_ch=TiffChannels.TWO,
                       verbose=True, cell_radius=4)
    c.find_colabeled()
    plt.show(block=True)
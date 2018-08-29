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
import warnings
import skimage.draw


class TiffChannels(enum.Enum):
    ONE = 0
    TWO = 1
    THREE = 2
    FOUR = 3


@attr.s
class ColabeledCells:
    """ 
    Analyze a TIF stack with two channels, one of them 
    shows activity and one is only morphological. The code
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
        self._find_cells(self.morph_img, self.struct_element)

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
        quantile_val = np.percentile(img, 90)
        binary_img = np.zeros_like(img)
        binary_img[img > quantile_val] = 1
        label, num_features = scipy.ndimage.label(binary_img, mask)
        if self.verbose:
            fig, ax = plt.subplots()
            ax.set_title(f'Cells (found {num_features} components)')
            ax.imshow(label)

        return scipy.ndimage.find_objects(label)


        
if __name__ == '__main__':
        c = ColabeledCells(tif=pathlib.Path('.'), result_file=pathlib.Path('..'),
                           activity_ch=TiffChannels.ONE, morph_ch=TiffChannels.TWO)
        print(c)
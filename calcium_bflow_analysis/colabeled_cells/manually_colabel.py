"""
This file aims at creating an object which allows the user
to manually curate the returned cells that CaImAn found
and create a list of indices that signify which cells out
of the field of cells CaImAn detected are "unique" (labeled).
This helps us connect data gathered in multiple spectral
channels.

The other file in the folder, "find_colabeled_cells.py",
does a similar job automatically. It works best when the
cells are round and full of fluorescence. When cells look
differently it's probably the easiest to use this manual
labeling technique.
"""
import pathlib
import warnings

import attr
from attr.validators import instance_of
import numpy as np
import matplotlib.pyplot as plt
import tifffile

from calcium_bflow_analysis.dff_analysis_and_plotting.plot_cells_and_traces import draw_rois_over_cells

@attr.s
class ManualLabeling:
    """
    Go over detected cells and manually figure out which
    cells are unique (i.e. labeled) and which are regular.
    Saves the output to disk as a list of indices corresponding
    to the cell indices that are unique.
    """
    tif = attr.ib(validator=instance_of(pathlib.Path))
    result_file = attr.ib(validator=instance_of(pathlib.Path))
    cell_radius = attr.ib(default=5, validator=instance_of(int))

    def run(self):
        """ Main pipeline """
        cell_idx = self._draw_rois_and_get_idx(self.tif, self.cell_radius)
        if cell_idx == [-1]:
            return
        self._serialize_idx(cell_idx)

    def _draw_rois_and_get_idx(self, tif, cell_radius):
        """
        Draws the FOV with the ROIS and receives from the user
        a list of cell indices which are the colabeled cells
        """
        draw_rois_over_cells(tif)
        plt.show(block=False)
        while True:
            idx_as_str = input(f'Tif: {tif}\nEnter colabeled cell indices as a list of numbers, separated by a comma. If none are colabeld, enter -1:\n')
            try:
                idx_as_str = idx_as_str.split(',')
                idx = [int(x) for x in idx_as_str]
                idx.sort()
                break
            except Exception as e:
                print(e)
                continue
        return idx

    def _serialize_idx(self, idx):
        """ Writes to disk the colabeled cell indices """
        fname = pathlib.Path(str(self.result_file)[:-11] + "colabeled_idx.npy")
        try:
            np.save(fname, idx)
        except PermissionError:
            warnings.warn(f"Permission error for folder {fname.parent}")


if __name__ == "__main__":
    tif = pathlib.Path('/data/Amit_QNAP/ForHagai/FOV2/355_GCaMP6-Ch2_WFA-590-Ch1_X25_mag3_act2-940nm_20180313_00003_CHANNEL_2.tif')
    result_file = pathlib.Path('/data/Amit_QNAP/ForHagai/FOV2/355_GCaMP6-Ch2_WFA-590-Ch1_X25_mag3_act2-940nm_20180313_00002_CHANNEL_3_results.npz')
    cell_radius = 14
    manu = ManualLabeling(tif=tif, result_file=result_file, cell_radius=cell_radius)
    manu.run()

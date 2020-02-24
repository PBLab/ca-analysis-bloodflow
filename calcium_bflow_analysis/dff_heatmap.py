import pathlib

import attr
from attr.validators import instance_of
import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.gridspec import GridSpec


@attr.s
class DffHeatmap:
    """
    TODO
    """
    # data_fname = attr.ib(validator=instance_of(str))
    caiman_results_folder = attr.ib(validator=instance_of(str))
    glob = attr.ib(default='*results.npz', validator=instance_of(str))
    files = attr.ib(init=False)
    dff = attr.ib(init=False)
    crd = attr.ib(init=False)
    valid_comps = attr.ib(init=False)
    comp_slices = attr.ib(init=False)

    def _find_files(self):
        self.files = pathlib.Path(self.caiman_results_folder).rglob(self.glob)
        first_fname = next(self.files)
        print(first_fname)
        first = np.load(first_fname)
        self.dff = first['F_dff']
        self.valid_comps = first['idx_components']
        self.crd = first['crd'][self.valid_comps]
        for file in self.files:
            print(file)
            caiman = np.load(file)
            self.dff = np.concatenate((self.dff, caiman['F_dff']))
            cur_valid =  caiman['idx_components']
            cur_crd = caiman['crd'][cur_valid]
            self.valid_comps = np.concatenate((self.valid_comps, cur_valid))
            self.crd = np.concatenate((self.crd, cur_crd))

    def display_dff(self):
        # self._compute_component_slices()
        self._find_files()
        self._display_heatmap()

    def _compute_component_slices(self):
        """ Run through all coordinates and extract the
        slice of their bounding box """
        self.comp_slices = pd.Series(dtype=object)
        for crd in self.crd:
            bbox = [int(val) for val in crd['bbox']]
            cur_slice = (slice(bbox[0], bbox[1]),
                         slice(bbox[2], bbox[3]))
            cur_neuron_id = crd['neuron_id']
            self.comp_slices.append(pd.Series({cur_neuron_id: cur_slice}, dtype=object))

    def _display_heatmap(self):
        fig, ax = plt.subplots()
        normed_dff = self.dff[::8, ::8].copy()
        normed_dff.flat[normed_dff.argmin()] = 0
        normed_dff -= normed_dff.min()
        normed_dff = np.log(normed_dff / normed_dff.max())
        ax.pcolor(normed_dff, vmin=normed_dff.min(), vmax=normed_dff.max())
        ax.set_aspect('auto')
        ax.set_ylabel('Cell ID')
        plt.show()


if __name__ == '__main__':
    DffHeatmap(r'X:\Amos\occluder').display_dff()
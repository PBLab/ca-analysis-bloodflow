"""
__author__ = Hagai Hargil
"""
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from enum import Enum
from typing import List
from itertools import chain
import peakutils
import scipy.spatial as spatial
from matplotlib.gridspec import GridSpec


class IterateOverCells(object):
    """
    Supports two different iteration functions.
    The first is to decide whether a components is a soma or dendrite. To use it, initiate an instance
    with a filename from CaImAn and the proper FPS, and call run with the parameters runtype='cells'.
    When it's running, press 'c' if it's a soma or 'd' if it's a dendrite. Press 'g' if you wish to
    discard this component.

    The second is to decide whether to merge or not a couple of close-by components. Run it by instantiating
    an instance of the class with the parameters 'ax_img' and 'ax_fluo' set to everything but None, and with the
    proper CaImAn result filename and FPS. To use it use the run method with the argument
    "runtype='merge'" and with the two arrays that are the result of the actions in CalciumData.merge_components().
     Press m if you wish to merge the two components shown.
    """
    def __init__(self, filename, fps, ax_img=None, ax_fluo=None):
        self.filename = filename
        self.fig = plt.figure()
        if ax_img is None:
            self.ax_img = self.fig.add_subplot(121)
        if ax_fluo is None:
            self.ax_fluo = self.fig.add_subplot(122)
        self.ax_fluo2 = None
        self.fps = fps
        self.Soma = namedtuple('Soma', ('x', 'y', 'idx'))
        self.Dendrite = namedtuple('Dendrite', ('x', 'y', 'idx'))
        self.soma_list = []
        self.dend_list = []
        self.merge_list = []
        self.global_idx = 0
        self.global_idx2 = 0

    def run(self, crdnts: np.ndarray, axis0: np.ndarray,
            axis1: np.ndarray, runtype: str='cells'):
        self.unpack_dict()
        if runtype == 'cells':
            self.iterate_over_cells_soma_dend()
            return None
        elif runtype == 'merge':
            self.iterate_over_cells_merge_comps(crdnts=crdnts, row_idx=axis0, col_idx=axis1)
            return self.merge_list

    def unpack_dict(self):
        self.full_dict = np.load(self.filename, encoding='bytes')
        try:
            self.fluo_trace = self.full_dict['Cdf']
        except KeyError:
            self.fluo_trace = self.full_dict['Cf']
        self.img_neuron = self.full_dict['Cn']

    def keypress_callback_soma_dend(self, event):
        """
        Handle key-presses for the iterate over cells function.
        :param event: event from matplotlib
        :return: list of indices of somas and of dendrites
        """

        if event.key == 'c':
            self.soma_list.append(self.global_idx)
        elif event.key == 'd':
            self.dend_list.append(self.global_idx)
        elif event.key == 'g':
            pass
        plt.close()

    def keypress_callback_merge(self, event):
        """
        Handle key-presses for merging double components
        :param event: event from matplotlib
        :return: list of pairs of indices to merge
        """
        if event.key == 'm':
            self.merge_list.append((self.global_idx, self.global_idx2))
        elif event.key == 'g':
            pass
        plt.close()

    def iterate_over_cells_soma_dend(self, key_callback=None):
        """
        Go through components and classify them whether they're a soma or dendrite using the keyboard
        :return:
        """
        if key_callback is None:
            key_callback = self.keypress_callback_soma_dend
        self.fig.canvas.mpl_connect('key_press_event', key_callback)
        self.ax_img.imshow(self.img_neuron, cmap='gray')
        self.ax_img.set_axis_off()
        for self.global_idx, item in enumerate(self.full_dict['crd']):
            self.redraw_soma_dend(item)
            fig_manager = plt.get_current_fig_manager()
            fig_manager.window.showMaximized()
            plt.show(block=True)
            # Reconnect
            self.fig = plt.figure()
            self.ax_img = self.fig.add_subplot(121)
            self.fig.canvas.mpl_connect('key_press_event', key_callback)
            self.ax_img.imshow(self.img_neuron, cmap='gray')
            self.ax_img.set_axis_off()
            self.ax_fluo = self.fig.add_subplot(122)

    def iterate_over_cells_merge_comps(self, row_idx: np.ndarray, col_idx: np.ndarray,
                                       crdnts: np.ndarray, key_callback=None):
        """
        Iterate of pairs of components to see if they're the same one
        :param key_callback:
        :param crdnts: Output from merge_components
        :param row_idx: Array of the row indices to iterate over
        :param col_idx: Array of the col indices
        :return:
        """
        if key_callback is None:
            key_callback = self.keypress_callback_merge
        self.fig.canvas.mpl_connect('key_press_event', key_callback)
        gs = GridSpec(2, 2)
        self.ax_img = plt.subplot(gs[:, 0])
        self.ax_img.set_axis_off()
        self.ax_fluo = plt.subplot(gs[0, 1])
        self.ax_fluo2 = plt.subplot(gs[1, 1])
        self.ax_img.imshow(self.img_neuron, cmap='gray')

        for self.global_idx, self.global_idx2 in zip(row_idx, col_idx):
            self.fig.suptitle(f'Current indices: {self.global_idx}, {self.global_idx2}')
            self.redraw_merge_components(crdnts)
            fig_manager = plt.get_current_fig_manager()
            fig_manager.window.showMaximized()
            plt.show(block=True)
            # Reconnect
            self.fig = plt.figure()
            gs = GridSpec(2, 2)
            self.ax_img = plt.subplot(gs[:, 0])
            self.ax_fluo = plt.subplot(gs[0, 1])
            self.ax_fluo2 = plt.subplot(gs[1, 1])
            self.fig.canvas.mpl_connect('key_press_event', key_callback)
            self.ax_img.imshow(self.img_neuron, cmap='gray')
            self.ax_img.set_axis_off()

    def redraw_soma_dend(self, item):
        cur_coor = item[b'coordinates']
        cur_coor = cur_coor[~np.isnan(cur_coor)].reshape((-1, 2))
        self.ax_img.plot(cur_coor[:, 0], cur_coor[:, 1])
        cur_trace = self.fluo_trace[self.global_idx, :]
        self.ax_fluo.plot(self.time_vec.T, cur_trace.T)
        self.ax_fluo.set_title(f"Index: {self.global_idx}")

    def redraw_merge_components(self, crdnts):
        st1 = self.ax_img.scatter(crdnts[self.global_idx][0], crdnts[self.global_idx][1],
                                  s=4, alpha=0.5, c='orange')
        st2 = self.ax_img.scatter(crdnts[self.global_idx2][0], crdnts[self.global_idx2][1],
                                  s=4, alpha=0.5, c='orange')
        self.ax_fluo.plot(self.time_vec.T, self.fluo_trace[self.global_idx, :])
        self.ax_fluo2.plot(self.time_vec.T, self.fluo_trace[self.global_idx2, :])

    @property
    def time_vec(self):
        return np.arange(start=0, stop=1/self.fps * (self.fluo_trace.shape[1]), step=1/self.fps)


class CalciumSource(Enum):
    SOMA = 'Cell bodies'
    DENDRITE = 'Dendrites'


class AcquisitionType(Enum):
    MULTISCALER = 'Multiscaler'
    ANALOG = 'Analog'


class CalciumData(object):
    def __init__(self, filename: str, cell_type: CalciumSource, acq_type: AcquisitionType,
                 idx: List, fps: float=15.24):
        self.filename = filename
        self.cell_type = cell_type
        self.acq_type = acq_type
        self.idx = idx
        self.fps = fps

    @property
    def all_data(self):
        return np.load(self.filename, encoding='bytes')

    @property
    def peak_widths(self):
        """
        Define a sequence of possible widths for a calcium peak
        :return: np.ndarray
        """
        MEAN_CALCIUM_PEAK_FWHM_IN_SECONDS = 0.9
        return np.arange(start=MEAN_CALCIUM_PEAK_FWHM_IN_SECONDS/3,
                         stop=MEAN_CALCIUM_PEAK_FWHM_IN_SECONDS*4,
                         step=MEAN_CALCIUM_PEAK_FWHM_IN_SECONDS/5,
                         dtype=np.float32) * self.fps

    def get_relevant_calcium_traces(self):
        """
        Grab only the rows that are in the specified index
        :return: np.array of traces, each row being a different CalciumSource
        """
        cur_traces = np.array([row_data for row_data in self.all_data['Cdf'][self.idx, :]])
        assert cur_traces.shape[0] == len(self.idx)
        return cur_traces

    def merge_components(self):
        """
        Find that belong to the same cell and merge them
        :return: Corrected list of indices
        """
        THRESHOLD = 20
        crdnts = self.all_data['crd'][self.idx]
        crdnts = np.array([item[b'CoM'] for item in crdnts])
        distances = spatial.distance.cdist(crdnts, crdnts, 'euclidean')
        triang = np.tril(distances)
        axis0, axis1 = np.nonzero(np.logical_and((triang < THRESHOLD), (triang > 0)))
        iterator = IterateOverCells(filename=self.filename, fps=self.fps,
                                    ax_img=True, ax_fluo=True)
        pairs_to_merge = iterator.run(runtype='merge', crdnts=crdnts, axis0=axis0,
                                      axis1=axis1)
        # Iterate over ROIs
        # pairs_to_merge = []
        # for idx, idy in zip(axis0, axis1):
        #     # Plot figure
        #     fig = plt.figure()
        #     ax = fig.add_subplot(111)
        #     ax.imshow(self.all_data['Cn'])
        #     st_1 = ax.scatter(crdnts[idx][0], crdnts[idx][1], s=4, alpha=0.7)
        #     st_2 = ax.scatter(crdnts[idy][0], crdnts[idy][1], s=4, alpha=0.7)
        #     plt.show(block=True)
        #     user_inp = input('Merge?')
        #     if user_inp == 'y':
        #         pairs_to_merge.append((idx, idy))
        #     st_1.remove()
        #     st_2.remove()
        return pairs_to_merge

    def discard_double_components(self):
        """
        Find components that are too close by and removes them
        """
        pairs = self.merge_components()
        max_idx_to_remove = set([max(item) for item in pairs])
        all_idx = np.array(self.idx)
        self.idx = list(np.delete(all_idx, max_idx_to_remove))
        print(f"Unmerged indices:\n{self.idx}")


class AnalyzeCalciumTraces(object):
    def __init__(self, data: CalciumData):
        self.calcium_data = data
        self.traces = data.get_relevant_calcium_traces()
        self.peaks = []

    def spike_amp_distrib(self) -> List:
        """
        Run the finds_peaks_cwt algorithm from SciPy on the filtered calcium traces from CaImAn.
        :return: List of peaks per row
        """

        for idx, row in enumerate(self.traces):
            self.peaks.append(
                peakutils.indexes(row, thres=0.4, min_dist=int(self.calcium_data.fps))
            )
        return self.peaks

    def visualize_peaks(self, indices: slice=slice(None, None, None)):
        """
        Place a mark wherever the algorithm found a calcium spike
        :param idx: Cells to plot
        :return: None
        """
        colors = [f"C{idx}" for idx in range(10)] * 10
        self.spike_amp_distrib()
        plt.figure()
        for idx, (cur_peaks, cur_trace) in enumerate(zip(self.peaks[indices], self.traces[indices, :])):
            plt.plot(cur_trace, colors[idx])
            plt.scatter(cur_peaks, cur_trace[cur_peaks], edgecolors=colors[idx])

    def histogram_peaks(self):
        """
        Create a histogram of the values of the peaks of the calcium trace
        :return: Flat list of peak values
        """
        self.spike_amp_distrib()
        vals = []
        for cur_peaks, cur_trace in zip(self.peaks, self.traces):
            vals.append(cur_trace[cur_peaks])
        vals_flat = list(chain.from_iterable(vals))
        plt.hist(vals_flat, bins=30)
        return vals_flat


if __name__ == '__main__':
    fps = 7.62
    fname = r'X:\Hagai\Multiscaler\27-9-17\For article\Calcium\FOV1_fromSI0000_d1_1024_d2_1024_d3_1_order_C_frames_1000_.results_analysis.npz'
    # iter = IterateOverCells(fname, fps)
    # iter.run()
    idx_soma_si = [4, 5, 6, 7, 12, 14, 15, 16, 24, 25, 30, 32, 33, 34, 44, 45, 46, 47, 51, 52, 54, 55, 56, 57, 58, 59, 62, 64, 65, 66, 67, 68, 69, 70, 71, 73, 79, 80, 82, 83, 85, 86, 87, 88, 91, 92, 94, 96, 99, 103, 105, 107, 108, 110, 112, 113, 115, 117, 118, 119, 120]
    cur_data = CalciumData(filename=fname, cell_type=CalciumSource.SOMA,
                           acq_type=AcquisitionType.ANALOG, idx=idx_soma_si,
                           fps=fps)
    cur_data.discard_double_components()
    # analyzed_data = AnalyzeCalciumTraces(cur_data)
    # peak_values = analyzed_data.histogram_peaks()
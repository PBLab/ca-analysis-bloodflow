import attr
from attr.validators import instance_of
import numpy as np
import pandas as pd
import pathlib
import sys
import peakutils
from statsmodels.stats.multicomp import MultiComparison
from statsmodels.stats.libqsturng import psturng
import scipy.stats
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib import gridspec
import mne
import tifffile
import os
import itertools
import xarray as xr
from collections import namedtuple
from datetime import datetime
import colorama
colorama.init()
from ansimarkup import ansiprint as aprint
import copy
import warnings

from ca_analysis.analog_trace import AnalogTraceAnalyzer
from ca_analysis.dff_tools import calc_dff, calc_dff_batch, scatter_spikes, plot_mean_vals, display_heatmap
from ca_analysis.vasc_occ_parsing import concat_vasc_occ_dataarrays



@attr.s
class VascOccAnalyzer:
    """ 
    Reads vascular occluder data from serialzed data and runs
    analysis methods on the it. If given more than one folder to
    look for files, it will concatenate all found files into a
    single large DataArray before the analysis.
    """
    folder_and_file = attr.ib(validator=instance_of(dict))
    with_analog = attr.ib(default=True, validator=instance_of(bool))
    data = attr.ib(init=False)
    dff = attr.ib(init=False)
    colabel_idx = attr.ib(init=False)
    split_data = attr.ib(init=False)
    all_spikes = attr.ib(init=False)
    labeled_cells = attr.ib(init=False)
    unlabeled_cells = attr.ib(init=False)

    def run_extra_analysis(self, dff, title: str='All cells'):
        """ Wrapper method to run several consecutive analysis scripts
        that all rely on a single dF/F matrix as their input """
        self.data = self._concat_dataarrays()
        all_spikes, num_peaks = self._find_spikes(self.epochs)
        self._calc_firing_rate(num_peaks, title)
        self._scatter_spikes(dff, all_spikes, title)
        self._rolling_window(dff, all_spikes, title)
        self._per_cell_analysis(num_peaks, title)
        if not self.with_analog:
            downsample_factor = 1 if title == 'Labeled' else 6
            display_heatmap(data=dff, epoch=title, downsample_factor=downsample_factor, 
                            fps=self.data.attrs['fps'])

        return all_spikes, num_peaks

    def _concat_dataarrays(self):
        """ Performs the concatenation of all given DataArrays
        into a single one before processing """
        all_da = []
        for folder, globstr in self.folder_and_file.items():
            all_da.append(xr.open_dataarray(str(next(folder.glob(globstr)))).load())
        return concat_vasc_occ_dataarrays(all_da)

    def _find_spikes(self, epochs: list):
        """ Calculates a dataframe, each row being a cell, with three columns - before, during and after
        the occlusion. The numbers for each cell are normalized for the length of the epoch."""
        idx_section1 = []
        idx_section2 = []
        idx_section3 = []
        thresh = 0.85
        min_dist = int(self.data.attrs['fps'])

        for epoch in epochs:
            dff_before = self.data.loc[{'epoch': epoch + '_before_occ'}].values

        all_spikes = np.zeros_like(dff)
        after_stim = self.frames_before_stim + self.len_of_epoch_in_frames
        norm_factor_during = self.frames_before_stim / self.len_of_epoch_in_frames
        norm_factor_after = self.frames_before_stim / self.frames_after_stim
        for row, cell in enumerate(dff):
            idx = peakutils.indexes(cell, thres=thresh, min_dist=min_dist)
            all_spikes[row, idx] = 1
            idx_section1.append(len(idx[idx < self.frames_before_stim]))
            idx_section2.append(len(idx[(idx >= self.frames_before_stim) &
                                        (idx < after_stim)]) * norm_factor_during)
            idx_section3.append(len(idx[idx >= after_stim]) * norm_factor_after)

        df = pd.DataFrame({'before': idx_section1, 'during': idx_section2, 'after': idx_section3},
                          index=np.arange(len(idx_section1)))
        return all_spikes, df

    def _calc_firing_rate(self, num_peaks: pd.DataFrame, epoch: str='All cells'):
        """
        Sum all indices of peaks to find the average firing rate of cells in the three epochs
        :return:
        """
        # Remove silent cells from comparison
        num_peaks.drop(self.invalid_cells, inplace=True)
        split_data = num_peaks.stack()
        mc = MultiComparison(split_data.values, split_data.index.get_level_values(1).values)
        try:
            res = mc.tukeyhsd()
        except ValueError:
            aprint("<yellow>Failed during the p-value calculation.</yellow>")
        else:
            print(res)
            print(f"P-values ({epoch}, number of cells: {split_data.shape[0] // 3}):", 
                  psturng(np.abs(res.meandiffs / res.std_pairs), 
                          len(res.groupsunique), 
                          res.df_total))
        finally:
            print(split_data.mean(level=1))

    def _scatter_spikes(self, dff, all_spikes, title='All cells'):
        """
        Show a scatter plot of spikes in the three epochs
        :param dff: Numpy array of cells x dF/F values
        :param all_spikes: DataFrame with number of spikes per trial.
        :return:
        """
        time = np.linspace(0, dff.shape[1]/self.fps, num=dff.shape[1], dtype=np.int32)
        fig, num_displayed_cells = scatter_spikes(dff, all_spikes, time_vec=time)
        ax = fig.axes[0]
        p = patches.Rectangle((self.frames_before_stim / self.fps, 0), width=self.len_of_epoch_in_frames / self.fps,
                              height=num_displayed_cells,
                              facecolor='red', alpha=0.3, edgecolor='None')
        ax.add_artist(p)
        ax.set_title(f'Scatter plot of spikes for cells: {title}')
        plt.savefig(f'spike_scatter_{title}.pdf', transparent=True)

    def _rolling_window(self, dff, all_spikes, epoch='All cells'):
        x_axis = np.arange(all_spikes.shape[1])/self.fps
        window = int(self.fps)
        fig_title = 'Rolling mean in epoch {epoch} over {over} ({win:.2f} sec window length)'

        ax_spikes, mean_val_spikes = plot_mean_vals(all_spikes, x_axis=x_axis, window=window, 
                                                    title=fig_title.format(epoch=epoch, 
                                                                           over='spike rate', 
                                                                           win=window/self.fps))
        ax_spikes.set_xlabel('Time (sec)')
        ax_spikes.set_ylabel('Mean Spike Rate')
        ax_spikes.plot(np.arange(self.frames_before_stim, self.frames_before_stim + self.len_of_epoch_in_frames)/self.fps,
                       np.full(self.len_of_epoch_in_frames, mean_val_spikes*3), 'r')
        plt.savefig('mean_spike_rate.pdf', transparent=True)
        ax_dff, mean_val_dff = plot_mean_vals(dff, x_axis=x_axis, window=int(self.fps),
                                              title=fig_title.format(epoch=epoch, over='dF/F', 
                                                                     win=window/self.fps))
        ax_dff.set_xlabel('Time (sec)')
        ax_dff.set_ylabel('Mean dF/F')
        ax_dff.plot(np.arange(self.frames_before_stim, self.frames_before_stim + self.len_of_epoch_in_frames)/self.fps,
                    np.full(self.len_of_epoch_in_frames, mean_val_dff*3), 'r')
        plt.savefig(f'mean_dff_{epoch}.pdf', transparent=True)

    def _per_cell_analysis(self, spike_freq_df, title='All cells'):
        """ Obtain a mean firing rate of each cell before, during and after the occlusion. Find
        the cells that have a large variance between these epochs. """
        # Normalization
        spike_freq_df['before_normed'] = 1
        spike_freq_df['during_normed'] = spike_freq_df['during'] / spike_freq_df['before']
        spike_freq_df['after_normed'] = spike_freq_df['after'] / spike_freq_df['before']

        spike_freq_df['variance'] = spike_freq_df.loc[:, 'before':'after'].var(axis=1)
        spike_freq_df['var_normed'] = spike_freq_df.loc[:, 'before_normed':'after_normed'].var(axis=1)

        repeat = spike_freq_df.loc[:, 'before':'after'].replace([np.inf, -np.inf], np.nan).dropna().values
        result = mne.stats.f_mway_rm(repeat, [3])

        fig, ax = plt.subplots()
        ax.plot(spike_freq_df.loc[:, 'before':'after'].T, '-o')
        ax.set_title(f'Per-cell analysis of {title}')

    def _visualize_occ_with_analog_data(self, file: str, dff: np.ndarray, analog_data: AnalogTraceAnalyzer):
        """ Show a figure with the dF/F heatmap, analog traces and occluder timings """

        fig = plt.figure()
        gs = gridspec.GridSpec(8, 1)
        self.__display_analog_traces(plt.subplot(gs[4, :]),
                                     plt.subplot(gs[5, :]),
                                     plt.subplot(gs[6, :]),
                                     analog_data)
        display_heatmap(ax=plt.subplot(gs[:4, :]), data=dff, fps=self.fps, downsample_factor=1)
        self.__display_occluder(plt.subplot(gs[7, :]), dff.shape[1])
        fig.suptitle(f'{file}')
        fig.tight_layout()
        plt.show()

    def _display_analog_traces(self, ax_puff, ax_jux, ax_run, data: AnalogTraceAnalyzer):
        """ Show three Axes of the analog data """
        ax_puff.plot(data.stim_vec)
        ax_puff.invert_yaxis()
        ax_puff.set_ylabel('Direct air puff')
        ax_puff.set_xlabel('')
        ax_puff.set_xticks([])
        ax_jux.plot(data.juxta_vec)
        ax_jux.invert_yaxis()
        ax_jux.set_ylabel('Juxtaposed puff')
        ax_jux.set_xlabel('')
        ax_jux.set_xticks([])
        ax_run.plot(data.run_vec)
        ax_run.invert_yaxis()
        ax_run.set_ylabel('Run times')
        ax_run.set_xlabel('')
        ax_run.set_xticks([])

    def _display_occluder(self, ax, data_length):
        """ Show the occluder timings """
        occluder = np.zeros((data_length))
        occluder[self.frames_before_stim:self.frames_before_stim + self.len_of_epoch_in_frames] = 1
        time = np.arange(data_length) / self.fps
        ax.plot(time, occluder)
        ax.get_xaxis().set_ticks_position('top')

        ax.invert_yaxis()

        ax.set_ylabel('Artery occlusion')
        ax.set_xlabel('')

    def _load_dff(self):
        """ Loads the dF/F data from all found files """
        self.dff = []
        for _, row in self.data_files.iterrows():
            cur_data = np.load(row.caiman)['F_dff']
            self.dff.append(cur_data)
        self.dff = np.concatenate(self.dff)
        return self.dff
    
    def _load_colabeled_idx(self):
        """ Loads the indices of the colabeled cells from all found files """
        self.colabel_idx = []
        num_of_cells = 0
        for _, row in self.data_files.iterrows():
            cur_data = np.load(row.caiman)['F_dff']
            cur_idx = np.load(row.colabeled)
            cur_idx += num_of_cells
            self.colabel_idx.append(cur_idx)
            num_of_cells += cur_data.shape[0]

        self.colabel_idx = np.array(list(itertools.chain.from_iterable(self.colabel_idx)))
        return self.colabel_idx


if __name__ == '__main__':
    folder = '/export/home/pblab/data/David/Vascular occluder_ALL/Thy_1_gcampF_vasc_occ_311018/left_hemi_(cca_left_with_vascular_occ)/'
    glob = r'f*results.npz'
    assert pathlib.Path(folder).exists()
    frames_before_stim = 17484
    len_of_epoch_in_frames = 7000
    fps = 58.2
    invalid_cells: list = []
    with_analog = True
    num_of_channels = 2
    with_colabeling = False
    display_each_fov = False
    serialize = True
    vasc = VascOccParser(foldername=folder, glob=glob,
                         frames_before_stim=frames_before_stim,
                         len_of_epoch_in_frames=len_of_epoch_in_frames,
                         fps=fps, invalid_cells=invalid_cells,
                         with_analog=with_analog,
                         num_of_channels=num_of_channels,
                         with_colabeling=with_colabeling,
                         display_each_fov=display_each_fov,
                         serialize=serialize)
    vasc.run()
    plt.show(block=True)


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
import caiman_funcs_for_comparison
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

from analog_trace import AnalogTraceAnalyzer
from dff_tools import calc_dff, calc_dff_batch, scatter_spikes, plot_mean_vals, display_heatmap


@attr.s(slots=True)
class VascOccAnalysis:
    """
    A class that provides the analysis pipeline for stacks with vascular occluder. Meaning,
    Data acquired in a before-during-after scheme, where "during" is the perturbation done
    to the system, occlusion of an artery in this case. The class needs to know how many frames
    were acquired before the perturbation and how many were acquired during. It also needs 
    other metadata, such as the framerate, and the IDs of cells that the CaImAn pipeline
    accidentally labeled as active components. If the data contains analog recordings as well,
    of the mouse's movements and air puffs, they will be integrated into the analysis as well.
    If one of the data channels contains co-labeling with a different, usually morphological, 
    fluorophore indicating the cell type, it will be integrated as well.
    """
    foldername = attr.ib(validator=instance_of(str))
    glob = attr.ib(default='*results.npz', validator=instance_of(str))
    fps = attr.ib(default=15.24, validator=instance_of(float))
    frames_before_stim = attr.ib(default=1000)
    len_of_epoch_in_frames = attr.ib(default=1000)
    invalid_cells = attr.ib(factory=list, validator=instance_of(list))
    with_analog = attr.ib(default=False, validator=instance_of(bool))
    with_colabeling = attr.ib(default=False, validator=instance_of(bool))
    num_of_channels = attr.ib(default=2, validator=instance_of(int))
    dff = attr.ib(init=False)
    split_data = attr.ib(init=False)
    all_spikes = attr.ib(init=False)
    frames_after_stim = attr.ib(init=False)
    start_time = attr.ib(init=False)
    timestamps = attr.ib(init=False)
    sliced_fluo = attr.ib(init=False)
    OccMetadata = attr.ib(init=False)
    data_files = attr.ib(init=False)
    colabel_idx = attr.ib(init=False)

    def __attrs_post_init__(self):
        self.OccMetadata = namedtuple('OccMetadata', ['before', 'during', 'after'])

    def run(self):
        self.__find_all_files()
        self.__get_params()
        if self.with_analog:
            self.__run_with_analog()
            self.dff = self.sliced_fluo.loc['all'].values
            if self.with_colabeling:
                self.colabel_idx = self.__load_colabeled_idx()
        elif self.with_colabeling:
            self.dff = self.__load_dff()
            self.colabel_idx = self.__load_colabeled_idx()
        else:
            self.dff = calc_dff_batch(self.data_files['caiman'])
        
        self.all_spikes, _ = self.__run_extra_analysis(self.dff)
        if self.with_colabeling:
            idx_unlabeled = np.setdiff1d(np.arange(self.dff.shape[0]), self.colabel_idx)
            dff_unlabeled = self.dff[idx_unlabeled]
            dff_labeled = self.dff[self.colabel_idx]
            all_spikes_labeled, peaks_labeled = self.__run_extra_analysis(dff_labeled, 'Labeled')
            all_spikes_unlabeled, peaks_unlabeled = self.__run_extra_analysis(dff_unlabeled, 'Unlabeled')
        
    def __run_extra_analysis(self, dff, title: str='All cells'):
        """ Wrapper method to run several consecutive analysis scripts
        that all rely on a single dF/F matrix as their input """
        all_spikes, num_peaks = self.__find_spikes(dff)
        self.__calc_firing_rate(num_peaks, title)
        self.__scatter_spikes(dff, all_spikes, title)
        self.__rolling_window(dff, all_spikes, title)
        self.__per_cell_analysis(num_peaks, title)
        if not self.with_analog:
            downsample_factor = 1 if title == 'Labeled' else 6
            display_heatmap(data=dff, epoch=title, downsample_factor=downsample_factor, 
                            fps=self.fps)

        return all_spikes, num_peaks

    def __run_with_analog(self):
        """ Helper function to run sequentially all needed analysis of dF/F + Analog data """
        list_of_sliced_fluo = []  # we have to compare each file with its analog data, individually
        for idx, row in self.data_files.iterrows():
            dff = calc_dff((row['caiman']))
            analog_data = pd.read_table(row['analog'], header=None, 
                                        names=['stimulus', 'run'], index_col=False)
            occ_metadata = self.OccMetadata(self.frames_before_stim, self.len_of_epoch_in_frames,
                                            self.frames_after_stim)
            # TODO: ADD SUPPORT FOR COLABELING WITH ANALOG DATA
            analog_trace = AnalogTraceAnalyzer(row['caiman'], analog_data, framerate=self.fps,
                                                num_of_channels=self.num_of_channels,
                                                start_time=self.start_time,
                                                timestamps=self.timestamps,
                                                occluder=True, occ_metadata=occ_metadata)
            analog_trace.run()
            copied_trace = copy.deepcopy(analog_trace)  # for some reason,
            # multiplying the trace by the dff changes analog_trace. To overcome
            # this weird issue we're copying it.
            list_of_sliced_fluo.append(analog_trace * dff)  # overloaded __mul__
            self.__visualize_occ_with_analog_data(row['tif'], dff, copied_trace)
        self.sliced_fluo = xr.concat(list_of_sliced_fluo, dim='neuron')

    def __find_all_files(self):
        """
        Locate all fitting files in the folder - The correpsonding "sibling" files
        for the main TIF recording, like analog data recordings, etc.
        """
        self.data_files = pd.DataFrame([], columns=['caiman', 'tif', 'analog', 'colabeled'])
        folder = pathlib.Path(self.foldername)
        files = folder.rglob(self.glob)
        print("Found the following files:")
        for idx, file in enumerate(files):
            print(file)
            cur_file = os.path.splitext(str(file.name))[0][:-18]  # no "_CHANNEL_X_results"
            try:
                raw_tif = next(file.parent.glob(cur_file + '.tif'))
            except StopIteration:
                print(f"No corresponding Tiff found for file {cur_file}.")
                raw_tif = ''
            
            try:
                analog_file = next(file.parent.glob(cur_file + '_analog.txt'))  # no 
            except StopIteration:
                print(f"No corresponding analog data found for file {cur_file}.")
                analog_file = ''
            
            try:
                colabeled_file = next(file.parent.glob(cur_file + '*_colabeled*.npy'))
            except StopIteration:
                print(f"No corresponding colabeled channel found for file {cur_file}. Did you run 'batch_colabeled'?")
                colabeled_file = ''

            self.data_files = self.data_files.append(pd.DataFrame([[str(file), raw_tif, analog_file, colabeled_file]],
                                                                  columns=['caiman', 'tif', 'analog', 'colabeled'],
                                                                  index=[idx]))

    def __get_params(self):
        """ Get general stack parameters from the TiffFile object """
        try:
            print("Getting TIF parameters...")
            with tifffile.TiffFile(self.data_files['tif'][0]) as f:
                si_meta = f.scanimage_metadata
                self.fps = si_meta['FrameData']['SI.hRoiManager.scanFrameRate']
                self.num_of_channels = len(si_meta['FrameData']['SI.hChannels.channelsActive'])
                num_of_frames = len(f.pages) // self.num_of_channels
                self.frames_after_stim = num_of_frames - (self.frames_before_stim + self.len_of_epoch_in_frames)
                self.start_time = str(datetime.fromtimestamp(os.path.getmtime(self.data_files['tif'][0])))
                self.timestamps = np.arange(num_of_frames)
                print("Done without errors!")
        except TypeError:
            self.start_time = None
            self.timestamps = None
            self.frames_after_stim = 1000
            print("Unsuccessful in getting the parameters.")

    def __find_spikes(self, dff):
        """ Calculates a dataframe, each row being a cell, with three columns - before, during and after
        the occlusion. The numbers for each cell are normalized for the length of the epoch."""
        idx_section1 = []
        idx_section2 = []
        idx_section3 = []
        thresh = 0.75
        min_dist = int(self.fps)
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

    def __calc_firing_rate(self, num_peaks: pd.DataFrame, epoch: str='All cells'):
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

    def __scatter_spikes(self, dff, all_spikes, title='All cells'):
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

    def __rolling_window(self, dff, all_spikes, epoch='All cells'):
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

    def __per_cell_analysis(self, spike_freq_df, title='All cells'):
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

    def __visualize_occ_with_analog_data(self, file: str, dff: np.ndarray, analog_data: AnalogTraceAnalyzer):
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

    def __display_analog_traces(self, ax_puff, ax_jux, ax_run, data: AnalogTraceAnalyzer):
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

    def __display_occluder(self, ax, data_length):
        """ Show the occluder timings """
        occluder = np.zeros((data_length))
        occluder[self.frames_before_stim:self.frames_before_stim + self.len_of_epoch_in_frames] = 1
        time = np.arange(data_length) / self.fps
        ax.plot(time, occluder)
        ax.get_xaxis().set_ticks_position('top')

        ax.invert_yaxis()

        ax.set_ylabel('Artery occlusion')
        ax.set_xlabel('')

    def __load_dff(self):
        """ Loads the dF/F data from all found files """
        self.dff = []
        for _, row in self.data_files.iterrows():
            cur_data = np.load(row.caiman)['F_dff']
            self.dff.append(cur_data)
        self.dff = np.concatenate(self.dff)
        return self.dff
    
    def __load_colabeled_idx(self):
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
    folder = '/export/home/pblab/data/David/vip_td_gcamp_270818_muscle_only'
    glob = r'*results.npz'
    frames_before_stim = 4000
    len_of_epoch_in_frames = 4000
    fps = 58.2
    invalid_cells: list = []
    with_analog = True
    num_of_channels = 2
    with_colabeling = True
    vasc = VascOccAnalysis(foldername=folder, glob=glob,
                           frames_before_stim=frames_before_stim,
                           len_of_epoch_in_frames=len_of_epoch_in_frames, 
                           fps=fps, invalid_cells=invalid_cells, 
                           with_analog=with_analog, 
                           num_of_channels=num_of_channels,
                           with_colabeling=with_colabeling)
    vasc.run()
    plt.show(block=True)


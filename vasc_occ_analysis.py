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
import mne


@attr.s(slots=True)
class VascOccAnalysis:
    """
    A class that provides the analysis pipeline for stacks with vascular occluder. Meaning,
    Data acquired in a before-during-after scheme, where "during" is the perturbation done
    to the system, occlusion of an artery in this case. The class needs to know how many frames
    were acquired before the perturbation and how many were acquired during. It also needs 
    other metadata, such as the framerate, and the IDs of cells that the CaImAn pipeline
    accidently labeled as active components.
    """
    foldername = attr.ib(validator=instance_of(str))
    glob = attr.ib(default='*results.npz', validator=instance_of(str))
    fps = attr.ib(default=15.24, validator=instance_of(float))
    frames_before_stim = attr.ib(default=1000)
    len_of_epoch_in_frames = attr.ib(default=1000)
    invalid_cells = attr.ib(default=[], validator=instance_of(list))
    dff = attr.ib(init=False)
    all_mice = attr.ib(init=False)
    split_data = attr.ib(init=False)
    all_spikes = attr.ib(init=False)
    frames_after_stim = attr.ib(init=False)
    dff_filtered = attr.ib(init=False)

    def run(self):
        files = self.__find_all_files()
        self.dff = self.__calc_dff(files)
        num_peaks = self.__find_spikes()
        self.__calc_firing_rate(num_peaks)
        self.__scatter_spikes()
        self.__rolling_window()
        # self.__per_cell_analysis(num_peaks)
        return self.dff

    def __find_all_files(self):
        """
        Locate all fitting files in the folder
        """
        self.all_mice = []
        files = pathlib.Path(self.foldername).rglob(self.glob)
        print("Found the following files:")
        for file in files:
            print(file)
            self.all_mice.append(str(file))
        files = pathlib.Path(self.foldername).rglob(self.glob)
        return files

    def __calc_dff(self, files):
        # sys.path.append(r'/data/Hagai/Multiscaler/code_for_analysis')
        import caiman_funcs_for_comparison

        all_data = []
        for file in files:
            data = np.load(file)
            print(f"Analyzing {file}...")
            try:
                all_data.append(data['F_dff'])
            except KeyError:
                all_data.append(caiman_funcs_for_comparison.detrend_df_f_auto(data['A'], data['b'], data['C'],
                                                                          data['f'], data['YrA']))
        return np.concatenate(all_data)

    def __find_spikes(self):
        """ Calculates a dataframe, each row being a cell, with three columns - before, during and after
        the occlusion. The numbers for each cell are normalized for the length of the epoch."""
        idx_section1 = []
        idx_section2 = []
        idx_section3 = []
        thresh = 0.65
        min_dist = int(self.fps)
        self.all_spikes = np.zeros_like(self.dff)
        after_stim = self.frames_before_stim + self.len_of_epoch_in_frames
        norm_factor_during = self.frames_before_stim / self.len_of_epoch_in_frames
        self.frames_after_stim = self.dff.shape[1] - (self.frames_before_stim + self.len_of_epoch_in_frames)
        norm_factor_after = self.frames_before_stim / self.frames_after_stim
        for row, cell in enumerate(self.dff):
            idx = peakutils.indexes(cell, thres=thresh, min_dist=min_dist)
            self.all_spikes[row, idx] = 1
            idx_section1.append(len(idx[idx < self.frames_before_stim]))
            idx_section2.append(len(idx[(idx >= self.frames_before_stim) &
                                        (idx < after_stim)]) * norm_factor_during)
            idx_section3.append(len(idx[idx >= after_stim]) * norm_factor_after)

        df = pd.DataFrame({'before': idx_section1, 'during': idx_section2, 'after': idx_section3},
                          index=np.arange(len(idx_section1)))
        return df

    def __calc_firing_rate(self, num_peaks: pd.DataFrame):
        """
        Sum all indices of peaks to find the average firing rate of cells in the three epochs
        :return:
        """
        # Remove silent cells from comparison
        num_peaks.drop(self.invalid_cells, inplace=True)
        self.split_data = num_peaks.stack()
        mc = MultiComparison(self.split_data.values, self.split_data.index.get_level_values(1).values)
        res = mc.tukeyhsd()
        print(res)
        print("P-values:", psturng(np.abs(res.meandiffs / res.std_pairs), len(res.groupsunique), res.df_total))
        print(self.split_data.mean(level=1))

    def __scatter_spikes(self):
        """
        Show a scatter plot of spikes in the three epochs
        :param before:
        :param during:
        :param after:
        :return:
        """
        x, y = np.nonzero(self.all_spikes)
        fig, ax = plt.subplots()
        downsample_display = 10
        num_displayed_cells = self.dff.shape[0] // downsample_display
        time = np.linspace(0, self.dff.shape[1]/self.fps, num=self.dff.shape[1], dtype=np.int32)
        ax.plot(time,
                (self.dff[:-10:downsample_display] + np.arange(num_displayed_cells)[:, np.newaxis]).T)
        peakvals = self.dff * self.all_spikes
        peakvals[peakvals == 0] = np.nan
        ax.plot(time,
                (peakvals[:-10:downsample_display] + np.arange(num_displayed_cells)[:, np.newaxis]).T,
                'r.', linewidth=0.1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Cell ID')
        p = patches.Rectangle((self.frames_before_stim / self.fps, 0), width=self.len_of_epoch_in_frames / self.fps,
                              height=num_displayed_cells,
                              facecolor='red', alpha=0.3, edgecolor='None')
        ax.add_artist(p)
        plt.savefig('spike_scatter.pdf', transparent=True)

    def __rolling_window(self):
        mean_spike = pd.DataFrame(self.all_spikes.mean(axis=0))
        mean_spike['x'] = np.arange(mean_spike.shape[0])/self.fps
        ax = mean_spike.rolling(window=int(self.fps)).mean().plot(x='x')
        ax.set_xlabel('Time (sec)')
        ax.set_ylabel('Mean Spike Rate')
        ax.set_title('Rolling mean (0.91 sec window length)')
        ax.plot(np.arange(self.frames_before_stim, self.frames_before_stim + self.len_of_epoch_in_frames)/self.fps,
                np.full(self.len_of_epoch_in_frames, 0.01), 'r')
        plt.savefig('mean_spike_rate.pdf', transparent=True)

        mean_dff = pd.DataFrame(self.dff.mean(axis=0))
        mean_dff['x'] = mean_spike['x']
        ax = mean_dff.rolling(window=int(self.fps)).mean().plot(x='x')
        ax.set_xlabel('Time (sec)')
        ax.set_ylabel('Mean dF/F')
        ax.set_title('Rolling mean over dF/F (0.91 sec window length)')
        ax.plot(np.arange(self.frames_before_stim, self.frames_before_stim + self.len_of_epoch_in_frames)/self.fps,
                np.full(self.len_of_epoch_in_frames, 0.01), 'r')
        plt.savefig('mean_dff.pdf', transparent=True)

    def __per_cell_analysis(self, spike_freq_df):
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


if __name__ == '__main__':
    vasc = VascOccAnalysis(foldername=r'/data/Amos/occluder/',
                           glob=r'*results.npz', frames_before_stim=17484,
                           len_of_epoch_in_frames=7000, fps=58.28,
                           invalid_cells=[])
    vasc.run()
    plt.show(block=False)
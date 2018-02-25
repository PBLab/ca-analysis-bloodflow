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


@attr.s(slots=True)
class VascOccAnalysis:
    foldername = attr.ib(validator=instance_of(str))
    glob = attr.ib(default='*results.npz', validator=instance_of(str))
    fps = attr.ib(default=7.68, validator=instance_of(float))
    len_of_epoch_in_frames = attr.ib(default=1000)
    dff = attr.ib(init=False)
    all_mice = attr.ib(init=False)
    split_data = attr.ib(init=False)
    all_spikes = attr.ib(init=False)

    def run(self):
        files = self.__find_all_files()
        self.__calc_dff(files)
        before, during, after = self.__find_spikes()
        self.__calc_firing_rate(before, during, after)
        self.__scatter_spikes()
        self.__rolling_window()
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
        sys.path.append(r'/data/Hagai/Multiscaler/code_for_analysis')
        import caiman_funcs_for_comparison

        coords = {'mouse': self.all_mice, }
        all_data = []
        for file in files:
            data = np.load(file)
            print(f"Analyzing {file}...")
            all_data.append(caiman_funcs_for_comparison.detrend_df_f_auto(data['A'], data['b'], data['C'],
                                                                          data['f'], data['YrA']))
        self.dff = np.concatenate(all_data)

    def __find_spikes(self):
        idx_section1 = []
        idx_section2 = []
        idx_section3 = []
        thresh = 0.55
        min_dist = 3
        self.all_spikes = np.zeros_like(self.dff)

        for row, cell in enumerate(self.dff):
            idx1 = peakutils.indexes(cell[:self.len_of_epoch_in_frames], thres=thresh, min_dist=min_dist)
            idx2 = peakutils.indexes(cell[self.len_of_epoch_in_frames:2*self.len_of_epoch_in_frames],
                                     thres=thresh, min_dist=min_dist)
            idx3 = peakutils.indexes(cell[2*self.len_of_epoch_in_frames:], thres=thresh, min_dist=min_dist)
            idx_section1.append(idx1)
            idx_section2.append(idx2)
            idx_section3.append(idx3)
            idxs = np.concatenate((idx1,
                                   idx2 + self.len_of_epoch_in_frames,
                                   idx3 + (2*self.len_of_epoch_in_frames)))
            self.all_spikes[row, idxs] = 1

        return idx_section1, idx_section2, idx_section3,

    def __calc_firing_rate(self, idx_section1, idx_section2, idx_section3):
        """
        Sum all indices of peaks to find the average firing rate of cells in the three epochs
        :param idx_section1:
        :param idx_section2:
        :param idx_section3:
        :return:
        """
        df = pd.DataFrame(columns=['before', 'during', 'after'], index=np.arange(len(idx_section1)))
        df['before'] = [len(cell) for cell in idx_section1]
        df['during'] = [len(cell) for cell in idx_section2]
        df['after'] = [len(cell) for cell in idx_section3]
        self.split_data = df.stack()
        mc = MultiComparison(self.split_data.values, self.split_data.index.get_level_values(1).values)
        res = mc.tukeyhsd()
        print(res)
        print(psturng(np.abs(res.meandiffs / res.std_pairs), len(res.groupsunique), res.df_total))
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
        ax.scatter(y, x, s=0.1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    def __rolling_window(self):
        summed = pd.Series(self.all_spikes.sum(axis=0))
        plt.figure()
        df = pd.Series(self.dff.sum(axis=0))
        summed.rolling(window=int(self.fps)).mean().plot()


if __name__ == '__main__':
    vasc = VascOccAnalysis(r'/data/David/Vas_occ_new_200218', glob=r'LH*results.npz')
    vasc.run()
"""
__author__ = Hagai Har-Gil
Many analysis functions for dF/F. Main class is CalciumReview.
"""
import attr
from attr.validators import instance_of
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from enum import Enum
import pathlib
import re

import dff_tools
from dff_tools import calc_auc, calc_mean_dff


class Condition(Enum):
    HYPER = 'Hyper'
    HYPO = 'Hypo'


class AvailableFuncs(Enum):
    """ Allowed analysis functions that can be used with CalciumReview.
    The values of the enum variants are names of functions in dff_tools.py """
    AUC = 'calc_auc'
    MEAN = 'calc_mean_dff'
    SPIKERATE = 'calc_mean_spike_rate'


@attr.s
class CalciumReview:
    """
    Evaluate and analyze calcium data from TAC-like experiments.
    The attributes ending with `_data` are pd.DataFrames that
    contain the result of different function from dff_tools.py. If you wish
    to add a new function, first make sure that its output is
    compatible with that of existing functions, then add a new
    attribute to the class and a new variant to the enum,
    and finally patch the __attrs_post_init__ method to include this
    new attribute. Make sure to not change the order of the enum - add
    the function at the bottom of that list.
    """
    folder = attr.ib(validator=instance_of(pathlib.Path))
    glob = attr.ib(default=r'data_of_day_*.nc')
    files = attr.ib(init=False)
    days = attr.ib(init=False)
    df_columns = attr.ib(init=False)
    funcs_dict = attr.ib(init=False)
    raw_data = attr.ib(init=False)
    auc_data = attr.ib(init=False)
    mean_data = attr.ib(init=False)
    spike_data = attr.ib(init=False)
    
    def __attrs_post_init__(self):
        """
        Find all files and parsed days for the experiment, and (partially) load them
        into memory.
         """
        self.files = []
        self.raw_data = {}
        all_files = folder.rglob(self.glob)
        day_reg = re.compile(r'of_day_(\d+).nc')
        parsed_days = []
        print("Found the following files:")
        for file in all_files:
            print(file)
            self.files.append(file)
            day = int(day_reg.findall(file.name)[0])
            parsed_days.append(day)
            self.raw_data[day] = xr.open_dataarray(file)
        self.days = np.array(parsed_days)

        self.df_columns = ['hypo_mean', 'hypo_std', 'hyper_mean', 'hyper_std']
        self.auc_data = pd.DataFrame(columns=self.df_columns)
        self.mean_data = pd.DataFrame(columns=self.df_columns)
        self.spike_data = pd.DataFrame(columns=self.df_columns)
        # Map the function name to its corresponding DataFrame
        self.funcs_dict = {key: val for key, val in zip(AvailableFuncs.__members__.values(),
                                                        [self.auc_data,
                                                         self.mean_data,
                                                         self.spike_data])}

    def data_of_day(self, day: int, condition: Condition, epoch='spont'):
        """ A function used to retrieve the "raw" data of dF/F, in the form of
        cells x time, to the user. Supply a proper day, condition and epoch and receive a numpy array. """
        assert type(condition) == Condition
        try:
            unselected_data = self.raw_data[day]
        except KeyError:
            print(f"The day {day} is invalid. Valid days are {self.days}.")
        else:
            return self._filter_da(unselected_data, condition=condition, epoch=epoch)

    def apply_analysis_funcs(self, funcs: list, epoch: str):
        """ Call the list of methods given to save time and memory """
        for day, raw_datum in self.raw_data.items():
            print(f"Analyzing day {day}...")
            selected_hyper = self._filter_da(raw_datum, condition='Hyper', epoch=epoch)
            selected_hypo = self._filter_da(raw_datum, condition='Hypo', epoch=epoch)
            for func in funcs:
                ans_hyper, ans_hyper_std = getattr(dff_tools, func.value)(selected_hyper)
                ans_hypo, ans_hypo_std = getattr(dff_tools, func.value)(selected_hypo)
                df_dict = {col: data for col, data in zip(self.df_columns,
                                                          [ans_hypo, ans_hypo_std, ans_hyper, ans_hyper_std])}
                self.funcs_dict[func] = self.funcs_dict[func].append(pd.DataFrame(df_dict, index=[day]))

    @staticmethod
    def plot_df(df, title):
        """ Helper method to plot DataFrames """
        fig, ax = plt.subplots()
        ax.errorbar(df.index.values, df.hypo_mean, df.hypo_std, c='C0', label='Hypo', fmt='o')
        ax.errorbar(df.index.values, df.hyper_mean, df.hyper_std, c='C1', label='Hyper', fmt='o')
        ax.legend()
        ax.set_xticks(df.index.values)
        ax.set_xlabel('Days')
        ax.set_title(title)

    def _filter_da(self, data, condition, epoch):
        """ Filter a DataArray by the given condition and epoch.
         Returns a numpy array in the shape of cells x time """
        selected = data.sel(condition=condition, epoch=epoch, drop=True).values
        relevant_idx = np.where(np.isfinite(selected))
        num_of_cells = len(np.unique(relevant_idx[0]))  # first dim is "neuron"
        selected = selected[relevant_idx].reshape((num_of_cells, -1))
        return selected


@attr.s(slots=True)
class CalciumAnalyzer:
    """
    Advanced methods to work and analyze calcium trace
    """
    data = attr.ib(validator=instance_of(xr.DataArray))
    cond = attr.ib(default=Condition.HYPER, validator=instance_of(Condition))
    plot = attr.ib(default=True)
    colors = attr.ib(init=False)
    num_of_neurons = attr.ib(init=False)

    def __attrs_post_init__(self):
        self.num_of_neurons = self.data.loc['spont'].values.shape[0]
        self.colors = [f'C{num % 10}' for num in range(10)]
        if self.num_of_neurons > 10:
            self.colors += [(val, val, val) for val in np.linspace(0, 1, self.num_of_neurons-10)]

    def run_analysis(self):
        """
        Run all possible analysis steps on the fluorescence data
        :return:
        """
        if self.plot == True:
            self.__show_spikes()
            self.__calc_mean_trace()

    def __show_spikes(self):
        """
        Plot each epoch with its fluorescence. Traces without activity are converted to nans
        :return:
        """
        for idx, arr in enumerate(self.data):
            fig, ax = plt.subplots()
            cur_tag = str(arr['epoch'].values)
            cur_data = arr.values.T
            normed_data = cur_data / np.atleast_2d(cur_data.max(axis=0))
            ax.plot(arr['time'].values, normed_data, label=cur_tag)
            self.__fig_manipulation(fig, ax, title=f'Fluorescent Trace in Epoch {cur_tag}',
                                    xlabel='Time', ylabel='Normalized Fluorescence')

    def __calc_mean_trace(self):
        """
        Find the number of measurements to average from each epoch - this number is the minimal number of measurements
        done in a single epoch. For example, if only 100 frames exist with stim_juxta - then the function would only
        sample 100 frames from the rest of the epochs. These 100 samples will be an average of several samples in the
        epochs that allow it.
        The the function plots the average value of each neuron in an experiment, in an epoch.
        :return: None
        """

        min_entries, nonzeros_in_epoch = self.__find_min_entries()
        sampled_data_to_plot = self.__sample_epochs(min_entries, nonzeros_in_epoch)
        x_axis = self.data.epoch.values.tolist()
        for idx, (epoch, sampled) in enumerate(zip(self.data, sampled_data_to_plot)):
            fig_means, ax_means = plt.subplots()
            cur_tag = str(epoch.epoch.values)
            ax_means.violinplot(sampled.T, points=100, showmeans=True)

            # Add scatters
            x_scat = np.tile(np.arange(1, sampled.shape[0]+1, dtype=np.uint32), (sampled.shape[1], 1))
            ax_means.scatter(x_scat.ravel(), sampled.T.ravel(), s=0.2, c='k', alpha=0.5)
            self.__fig_manipulation(fig_means, ax_means, title=f'Mean Fluorescence Per Neuron, {cur_tag} Epoch',
                                    xlabel='Neuron Number', ylabel=r'$\Delta$F/F')

        fig_all, ax_all = plt.subplots()
        ax_all.violinplot(sampled_data_to_plot.mean(axis=-1).T,
                          points=100, showmeans=True)
        ax_all.set_xticks(np.arange(1, len(self.data)+1))
        ax_all.set_xticklabels(x_axis)
        self.__fig_manipulation(fig_all, ax_all, title=f'All Cells dF_F in {self.cond} Under All Epochs',
                                xlabel='Epoch', ylabel='Mean dF/F')

    def __find_min_entries(self) -> Tuple[int, Dict[str, Tuple]]:
        """
        Find the shortest epoch in the experiment
        :return: Smallest number of frames, and the number of frames of each epoch
        """
        min_entries = 1e6
        nonzeros_in_epoch = {}
        for arr in self.data:
            nonzeros = np.nonzero(arr.values.sum(axis=0))[0]
            nonzeros_in_epoch[str(arr.epoch.values)] = (nonzeros, nonzeros.shape[0])
            if 0 < nonzeros.shape[0] < min_entries:
                min_entries = nonzeros.shape[0]

        return min_entries, nonzeros_in_epoch

    def __sample_epochs(self, min_entries: int, nonzeros_in_epoch: Dict[str, Tuple]) -> np.ndarray:
        """
        For each epoch sample only min_entries frames from it. For the epochs that contain more than min_entries frames,
        average out as many samples as you can
        :param min_entries: Smallest number of samples in an epoch
        :param nonzeros_in_epoch: How many valid cells are in each epoch
        :return: Array of the mean fluorescence per neuron
        """
        data_per_neuron = np.zeros((self.data.shape[0], self.num_of_neurons, min_entries))
        for idx, arr in enumerate(self.data):
            key = str(arr.epoch.values)
            iters_of_sampling = nonzeros_in_epoch[key][1] // min_entries
            if iters_of_sampling > 0:
                sampled = np.zeros((iters_of_sampling, self.num_of_neurons, min_entries), dtype=np.uint32)
                for iter in range(iters_of_sampling):
                    rand_idx = np.random.choice(nonzeros_in_epoch[key][0],
                                                min_entries).astype(np.uint32)
                    sampled[iter, ...] = arr.values[:, rand_idx]
                data_per_neuron[idx, :] = sampled.sum(axis=0) / iters_of_sampling

        return data_per_neuron

    def __fig_manipulation(self, fig: plt.Figure, ax: plt.Axes, title: str, xlabel: str, ylabel: str):
        """
        General figure manipulations and savings
        :param fig: Figure
        :param ax: Axes
        :param title:
        :param xlabel:
        :param ylabel:
        :return:
        """
        fig.suptitle(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        fig.patch.set_alpha(0)
        # ax.set_frame_on(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(0.5)
        ax.spines['left'].set_linewidth(0.5)
        try:
            plt.savefig(title+'.pdf', format='pdf', transparent=True, dpi=1000)
        except PermissionError:
            print("Couldn't save figure due to a permission error.")

if __name__ == '__main__':
    folder = pathlib.Path.home() / pathlib.Path(r'data/David/crystal_skull_TAC_180719')
    assert folder.exists()
    ca = CalciumReview(folder)
    ca.apply_analysis_funcs([AvailableFuncs.AUC, AvailableFuncs.MEAN,
                             AvailableFuncs.SPIKERATE], 'spont')
    plt.show()
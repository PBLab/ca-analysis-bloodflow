"""
__author__ = Hagai Har-Gil
"""
import attr
from attr.validators import instance_of
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from enum import Enum


class Condition(Enum):
    HYPER = 'Hyper'
    HYPO = 'Hypo'


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
                                    xlabel='Neuron Number', ylabel='$\Delta$F/F')

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


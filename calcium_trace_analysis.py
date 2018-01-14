"""
__author__ = Hagai Har-Gil
"""
import attr
from attr.validators import instance_of
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from typing import List, Tuple


@attr.s(slots=True)
class CalciumAnalyzer:
    """
    Advanced methods to work and analyze calcium trace
    """
    data = attr.ib(validator=instance_of(xr.DataArray))
    colors = attr.ib(init=False)
    num_of_neurons = attr.ib(init=False)

    def __attrs_post_init__(self):
        self.colors = [f'C{num % 10}' for num in range(len(self.data))]
        self.num_of_neurons = self.data.loc['stim'].values.shape[0]

    def run_analysis(self):
        """
        Run all possible analysis steps on the fluorescence data
        :return:
        """
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
        fig_means, ax_means = plt.subplots()

        for idx, (arr, data) in enumerate(zip(self.data, sampled_data_to_plot)):
            cur_tag = str(arr['epoch'].values)
            ax_means.scatter(arr['neuron'].values, data, label=cur_tag, color=self.colors[idx])
            ax_means.plot(arr['neuron'].values, np.tile(data.mean(), self.num_of_neurons),
                          label=cur_tag+'_mean', c=self.colors[idx])

            plt.legend()
            self.__fig_manipulation(fig_means, ax_means, title='Mean Fluorescence Per Neuron',
                                    xlabel='Neuron Number', ylabel='Fluorescence')

    def __find_min_entries(self) -> Tuple[int, List[int]]:
        """
        Find the shortest epoch in the experiment
        :return: Smallest number of frames, and the number of frames of each epoch
        """
        min_entries = 1e6
        nonzeros_in_epoch = []
        for arr in self.data:
            num_positive = np.nonzero(arr.values.sum(axis=0))[0].shape[0]
            nonzeros_in_epoch.append(num_positive)
            if 0 < num_positive < min_entries:
                min_entries = num_positive

        return min_entries, nonzeros_in_epoch

    def __sample_epochs(self, min_entries: int, nonzeros_in_epoch: List[int]) -> np.ndarray:
        """
        For each epoch sample only min_entries frames from it. For the epochs that contain more than min_entries frames,
        average out as many samples as you can
        :param min_entries: Smallest number of samples in an epoch
        :param nonzeros_in_epoch: How many valid cells are in each epoch
        :return: Array of the mean fluorescence per neuron
        """
        data_per_neuron = np.zeros((self.data.shape[0], self.num_of_neurons))
        for idx, arr in enumerate(self.data):
            prelim_result = np.zeros((min_entries, self.num_of_neurons))
            iters_of_sampling = nonzeros_in_epoch[idx] // min_entries
            if iters_of_sampling > 0:
                sampled = np.zeros((iters_of_sampling, min_entries), dtype=np.uint32)
                for iter in range(iters_of_sampling):
                    sampled[iter, ...] = np.random.choice(nonzeros_in_epoch[idx], min_entries).astype(np.uint32)
                for sample in sampled:
                    prelim_result += arr[:, sample].values.T
                data_per_neuron[idx, :] = (prelim_result / iters_of_sampling).mean(axis=0)

        # mean_per_neuron = data_per_neuron / np.atleast_2d(data_per_neuron.max(axis=1))

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


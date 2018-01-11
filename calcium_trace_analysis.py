"""
__author__ = Hagai Har-Gil
"""
import attr
from attr.validators import instance_of
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

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
            cur_tag = str(arr['sig_type'].values)
            cur_data = arr.values.T
            normed_data = cur_data / np.atleast_2d(cur_data.max(axis=0))
            ax.plot(arr['time'].values, normed_data, label=cur_tag)
            self.__fig_manipulation(fig, ax, title=f'Fluorescent Trace in Epoch {cur_tag}',
                                    xlabel='Time', ylabel='Normalized Fluorescence')


    def __calc_mean_trace(self):
        """
        Plot the average value of each neuron in an experiment
        :return:
        """
        fig_means, ax_means = plt.subplots()

        for idx, arr in enumerate(self.data):
            cur_tag = str(arr['sig_type'].values)
            normed_data = arr.values.T / np.atleast_2d(arr.values.T.max(axis=0))
            means = normed_data.mean(axis=0)
            ax_means.scatter(arr['neuron'].values, means, label=cur_tag, color=self.colors[idx])
            ax_means.plot(arr['neuron'].values, np.tile(means.mean(), self.num_of_neurons),
                          label=cur_tag+'_mean', c=self.colors[idx])

        plt.legend()
        self.__fig_manipulation(fig_means, ax_means, title='Mean Fluorescence Per Neuron',
                                xlabel='Neuron Number', ylabel='Fluorescence')


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


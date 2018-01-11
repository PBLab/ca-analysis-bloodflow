"""
__author__ = Hagai Har-Gil
"""
import attr
from attr.validators import instance_of
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt


class CalciumAnalyzer:
    """
    Advanced methods to work and analyze calcium trace
    """
    data = attr.ib(validator=instance_of(xr.DataArray))

    def run_analysis(self):
        """
        Run all possible analysis steps on the fluorescence data
        :return:
        """
        self.__calc_mean_trace()

    def __calc_mean_trace(self):
        """
        Plot the average value of each neuron in an experiment
        :return:
        """
        fig_means, ax_means = plt.subplots()

        for arr in self.data:
            print(1)
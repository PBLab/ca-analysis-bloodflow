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
import itertools
from scipy import stats

import dff_tools


class Condition(Enum):
    HYPER = 'HYPER'
    HYPO = 'HYPO'


class AvailableFuncs(Enum):
    """ Allowed analysis functions that can be used with CalciumReview.
    The values of the enum variants are names of functions in dff_tools.py """
    AUC = 'calc_auc'
    MEAN = 'calc_mean_dff'
    SPIKERATE = 'calc_mean_spike_num'


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
    conditions = attr.ib(init=False)
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
        stats = ['_mean', '_std']
        self.conditions = self.raw_data[day].condition.values.tolist()
        self.df_columns = [''.join(x) for x in itertools.product(self.conditions, stats)] + ['t', 'p']
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
            selected_first = self._filter_da(raw_datum, condition=self.conditions[0], epoch=epoch)
            selected_second = self._filter_da(raw_datum, condition=self.conditions[1], epoch=epoch)
            for func in funcs:
                cond1 = getattr(dff_tools, func.value)(selected_first)
                cond1_mean, cond1_sem = cond1.mean(), cond1.std(ddof=1) / np.sqrt(cond1.shape[0]) 
                cond2 = getattr(dff_tools, func.value)(selected_second)
                cond2_mean, cond2_sem = cond2.mean(), cond2.std(ddof=1) / np.sqrt(cond2.shape[0])
                t, p = stats.ttest_ind(cond1, cond2, equal_var=False)
                df_dict = {col: data for col, data in zip(self.df_columns,
                    [cond1_mean, cond1_sem, cond2_mean, cond2_sem, t, p])}
                self.funcs_dict[func] = self.funcs_dict[func].append(pd.DataFrame(df_dict, index=[day]))

    def plot_df(self, df, title):
        """ Helper method to plot DataFrames """
        fig, ax = plt.subplots()
        ax.errorbar(df.index.values, df[df.columns[0]], df[df.columns[1]], 
                    c='C0', label=self.conditions[0], fmt='o')
        ax.errorbar(df.index.values, df[df.columns[2]], df[df.columns[3]], 
                    c='C1', label=self.conditions[1], fmt='o')
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


if __name__ == '__main__':
    folder = pathlib.Path.home() / pathlib.Path(r'data/David/crystal_skull_TAC_180719')
    assert folder.exists()
    ca = CalciumReview(folder)
    analysis_methods = [AvailableFuncs.AUC, AvailableFuncs.MEAN,
                        AvailableFuncs.SPIKERATE]
    epoch = 'spont'
    ca.apply_analysis_funcs(analysis_methods, epoch)
    plt.show()
"""
__author__ = Hagai Har-Gil
Many analysis functions for dF/F. Main class is CalciumReview.
"""
import pathlib
import re
import itertools
import warnings
from typing import Optional, Dict

import attr
from attr.validators import instance_of
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from enum import Enum
from scipy import stats

from calcium_bflow_analysis.dff_analysis_and_plotting import dff_analysis
from calcium_bflow_analysis.single_fov_analysis import filter_da


class Condition(Enum):
    HYPER = "HYPER"
    HYPO = "HYPO"


class AvailableFuncs(Enum):
    """ Allowed analysis functions that can be used with CalciumReview.
    The values of the enum variants are names of functions in dff_analysis.py """

    AUC = "calc_total_auc_around_spikes"
    MEAN = "calc_mean_auc_around_spikes"
    SPIKERATE = "calc_mean_spike_num"


@attr.s
class CalciumReview:
    """
    Evaluate and analyze calcium data from TAC-like experiments.
    The attributes ending with `_data` are pd.DataFrames that
    contain the result of different function from dff_analysis.py. If you wish
    to add a new function, first make sure that its output is
    compatible with that of existing functions, then add a new
    attribute to the class and a new variant to the enum,
    and finally patch the __attrs_post_init__ method to include this
    new attribute. Make sure to not change the order of the enum - add
    the function at the bottom of that list.
    """

    folder = attr.ib(validator=instance_of(pathlib.Path))
    glob = attr.ib(default=r"*data_of_day_*.nc")
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
        all_files = self.folder.rglob(self.glob)
        day_reg = re.compile(r".+?of_day_(\d+).nc")
        parsed_days = []
        print("Found the following files:")
        for file in all_files:
            print(file)
            self.files.append(file)
            try:
                day = int(day_reg.findall(file.name)[0])
            except IndexError:
                continue
            parsed_days.append(day)
            self.raw_data[day] = xr.open_dataset(file)
        self.days = np.unique(np.array(parsed_days))
        stats = ["_mean", "_std"]
        self.conditions = list(set(self.raw_data[day].condition.values.tolist()))
        self.df_columns = [
            "".join(x) for x in itertools.product(self.conditions, stats)
        ] + ["t", "p"]
        self.auc_data = pd.DataFrame(columns=self.df_columns)
        self.mean_data = pd.DataFrame(columns=self.df_columns)
        self.spike_data = pd.DataFrame(columns=self.df_columns)
        # Map the function name to its corresponding DataFrame
        self.funcs_dict = {
            key: val
            for key, val in zip(
                AvailableFuncs.__members__.values(),
                [self.auc_data, self.mean_data, self.spike_data],
            )
        }

    def data_of_day(self, day: int, condition: Condition, epoch="spont"):
        """ A function used to retrieve the "raw" data of dF/F, in the form of
        cells x time, to the user. Supply a proper day, condition and epoch and receive a numpy array. """
        try:
            unselected_data = self.raw_data[day]
        except KeyError:
            print(f"The day {day} is invalid. Valid days are {self.days}.")
        else:
            return filter_da(unselected_data, condition=condition.value, epoch=epoch)

    def apply_analysis_funcs_two_conditions(
        self, funcs: list, epoch: str, mouse_id: Optional[str] = None
    ):
        """ Call the list of methods given to save time and memory. Applicable
        if the dataset has two conditions, like left and right. """
        norm1, norm2 = 1, 1
        for day, raw_datum in dict(sorted(self.raw_data.items())).items():
            print(f"Analyzing day {day}...")
            selected_first = filter_da(
                raw_datum, condition=self.conditions[0], epoch=epoch, mouse_id=mouse_id,
            )
            selected_second = filter_da(
                raw_datum, condition=self.conditions[1], epoch=epoch, mouse_id=mouse_id,
            )
            if selected_first.shape[0] == 0 or selected_second.shape[0] == 0:
                continue
            for func in funcs:
                cond1 = getattr(dff_analysis, func.value)(selected_first, self.raw_data[day].fps)
                cond1_mean, cond1_sem = (
                    cond1.mean(),
                    cond1.std(ddof=1) / np.sqrt(cond1.shape[0]),
                )
                cond2 = getattr(dff_analysis, func.value)(selected_second, self.raw_data[day].fps)
                cond2_mean, cond2_sem = (
                    cond2.mean(),
                    cond2.std(ddof=1) / np.sqrt(cond2.shape[0]),
                )
                # if func == AvailableFuncs.AUC and day == 0:
                #     norm1 = cond1_mean
                #     norm2 = cond2_mean
                t, p = stats.ttest_ind(cond1, cond2, equal_var=False)
                df_dict = {
                    col: data
                    for col, data in zip(
                        self.df_columns,
                        [
                            cond1_mean / norm1,
                            cond1_sem / norm1,
                            cond2_mean / norm2,
                            cond2_sem / norm2,
                            t,
                            p,
                        ],
                    )
                }
                self.funcs_dict[func] = self.funcs_dict[func].append(
                    pd.DataFrame(df_dict, index=[day])
                )

    def apply_analysis_single_condition(
        self, funcs: list, epoch: str, mouse_id: Optional[str] = None
    ):
        """Run a list of methods on the object for the given epoch if we have
        only a single condition"""

        for day, raw_datum in dict(sorted(self.raw_data.items())).items():
            print(f"Analyzing day {day}...")
            selected_first = filter_da(
                raw_datum, condition=self.conditions[0], epoch=epoch, mouse_id=mouse_id
            )
            if selected_first.shape[0] == 0:
                warnings.warn("No data rows in this day.")
                continue
            for func in funcs:
                cond1 = getattr(dff_analysis, func.value)(selected_first, self.raw_data[day].fps)
                cond1_mean, cond1_sem = (
                    cond1.mean(),
                    cond1.std(ddof=1) / np.sqrt(cond1.shape[0]),
                )
                df_dict = {
                    col: data
                    for col, data in zip(
                        [self.df_columns[0], self.df_columns[1]],
                        [cond1_mean, cond1_sem],
                    )
                }
                self.funcs_dict[func] = self.funcs_dict[func].append(
                    pd.DataFrame(df_dict, index=[day])
                )

    def plot_df_two_conditions(self, df, title, conditions=None):
        """ Helper method to plot DataFrames """
        if conditions is None:
            conditions = self.conditions
        fig, ax = plt.subplots()

        ax.errorbar(
            df.index.values,
            df[df.columns[0]],
            df[df.columns[1]],
            c="C0",
            label=conditions[0],
            fmt="-o",
        )
        ax.errorbar(
            df.index.values,
            df[df.columns[2]],
            df[df.columns[3]],
            c="C1",
            label=conditions[1],
            fmt="-o",
        )
        ax.legend()
        ax.set_xticks(df.index.values)
        ax.set_xlabel("Days")
        ax.set_title(title)

    def plot_single_condition(self, df, title):
        fig, ax = plt.subplots()
        ax.errorbar(df.index.values, df.iloc[:, 0], df.iloc[:, 1], c="C0", fmt="o")
        ax.set_xticks(df.index.values)
        ax.set_xlabel("Days")
        ax.set_title(title)


def plot_single_cond_per_mouse(ca: CalciumReview, analysis_methods: list):
    """Generates a plot of the values over time of each mouse in the experiment"""
    mids = np.unique(ca.raw_data[0].mouse_id.values)
    stats = ["mean", "std"]
    columns = ["_".join(x) for x in itertools.product(mids, stats)]
    datacache = pd.DataFrame(index=sorted(list(ca.raw_data.keys())), columns=columns)
    results = {func_name.name: datacache.copy() for func_name in analysis_methods}
    for day, raw_datum in dict(sorted(ca.raw_data.items())).items():
        print(f"Analyzing day {day}...")
        for mid, data in raw_datum.groupby(raw_datum.mouse_id):
            filtered = filter_da(data, epoch="all")
            for func in analysis_methods:
                result = getattr(dff_analysis, func.value)(filtered)
                result_mean, result_sem = (
                    result.mean(),
                    result.std(ddof=1) / np.sqrt(result.shape[0]),
                )
                results[func.name].loc[day, f"{mid}_mean"] = result_mean
                results[func.name].loc[day, f"{mid}_std"] = result_sem

    ca.plot_df_two_conditions(results["AUC"], "AUC both mice", mids)
    ca.plot_df_two_conditions(results["SPIKERATE"], "Spike rate both mice", mids)


if __name__ == "__main__":
    folder = pathlib.Path(r"/data/David/TAC_group_7_baseline")
    assert folder.exists()
    ca = CalciumReview(folder, "data_*.nc")
    analysis_methods = [
        AvailableFuncs.AUC,
        AvailableFuncs.MEAN,
        AvailableFuncs.SPIKERATE,
    ]
    epoch = "all"
    ca.apply_analysis_funcs_two_conditions(analysis_methods, epoch)
    ca.plot_df_two_conditions(
        ca.funcs_dict[AvailableFuncs.AUC], f"AUC of Fluo Traces, Epoch: {epoch}"
    )
    ca.plot_df_two_conditions(
        ca.funcs_dict[AvailableFuncs.SPIKERATE],
        f"Spike Rate of Fluo Traces, Epoch: {epoch}",
    )
    ca.plot_df_two_conditions(
        ca.funcs_dict[AvailableFuncs.MEAN], f"Mean dF/F of Fluo Traces, Epoch: {epoch}"
    )
    # ca.apply_analysis_single_condition(analysis_methods, epoch, mouse_id='514')
    # ca.plot_single_condition(ca.funcs_dict[AvailableFuncs.AUC], f"AUC of Fluo Traces, Epoch: {epoch} [514]")
    # ca.plot_single_condition(ca.funcs_dict[AvailableFuncs.SPIKERATE], f"Spike Rate of Fluo Traces, Epoch: {epoch} [514]")
    # plot_single_cond_per_mouse(ca, analysis_methods)
    plt.show(block=False)

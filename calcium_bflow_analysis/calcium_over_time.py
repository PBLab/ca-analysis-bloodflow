"""
A module designed to analyze FOVs of in vivo calcium
activity. This module's main class, :class:`CalciumOverTime`,
is used to run
"""

from enum import Enum
from pathlib import Path
import os
import re
from collections import defaultdict
import itertools
from datetime import datetime
import multiprocessing as mp
from typing import Tuple, List, Optional

import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import attr
import tifffile
from scipy.ndimage.morphology import binary_fill_holes
from attr.validators import instance_of

from calcium_bflow_analysis.fluo_metadata import FluoMetadata
from calcium_bflow_analysis.analog_trace import AnalogAcquisitionType
from calcium_bflow_analysis.trace_converter import RawTraceConverter, ConversionMethod
import calcium_bflow_analysis.caiman_funcs_for_comparison
from calcium_bflow_analysis.single_fov_analysis import SingleFovParser
from calcium_bflow_analysis.calcium_trace_analysis import Condition


class Epoch(Enum):
    """
    All possible TAC epoch combinations
    """

    ALL = "all"
    RUN = "run"
    STAND = "stand"
    STIM = "stim"
    JUXTA = "juxta"
    SPONT = "spont"
    RUN_STIM = "run_stim"
    RUN_JUXTA = "run_juxta"
    RUN_SPONT = "run_spont"
    STAND_STIM = "stand_stim"
    STAND_JUXTA = "stand_juxta"
    STAND_SPONT = "stand_spont"


@attr.s(slots=True)
class FileFinder:
    """
    A class designated to find all doublets or triplets of files
    for a given FOV. This means that each tif file that represents
    a recorded field of view should always have a corresponding
    .npz file containing the results from the calcium analysis
    pipeline, and if analog data was recorded then this FOV also
    has a .txt file to go along with it. This class is aimed at
    finding these triplets (or doublets if the analog file is
    missing) and returning them to other classes for furthing
    processing.
    """

    results_folder = attr.ib(validator=instance_of(Path))
    folder_globs = attr.ib(default={Path("."): "*.tif"}, validator=instance_of(dict))
    analog = attr.ib(default=AnalogAcquisitionType.NONE)
    with_colabeled = attr.ib(default=False, validator=instance_of(bool))
    data_files = attr.ib(init=False)

    def find_files(self) -> pd.DataFrame:
        """
        Main entrance to pipeline of class. Returns a DataFrame in which
        each row is a doublet\\triplet of corresponding files.
        """
        fluo_files, analog_files, result_files, colabeled_files = (
            self._find_all_relevant_files()
        )
        self.data_files = self._make_table(
            fluo_files, analog_files, result_files, colabeled_files
        )
        return self.data_files

    def _find_all_relevant_files(self) -> Tuple[List[Optional[Path]], ...]:
        """
        Passes each .tif file it finds (with the given glob string)
        and looks for its results, analog and colabeled friends.
        If it can't find the friends it skips this file, else it adds
        them into a list. A list None is returned if this
        experiment had no colabeling or analog data associated with it.
        """
        fluo_files = []
        analog_files = []
        result_files = []
        colabeled_files = []
        summary_str = "Found the following {num} files:\nFluo: {fluo}\nAnalog: {analog}\nCaImAn: {caiman}\nColabeled: {colabeled}"
        for folder, globstr in self.folder_globs.items():
            for file in folder.rglob(globstr):
                num_of_files_found = 1
                fname = str(file.name)[:-4]
                if self.analog is not AnalogAcquisitionType.NONE:
                    try:
                        analog_file = next(folder.rglob(fname + "*analog*.txt"))
                        num_of_files_found += 1
                    except StopIteration:
                        print(f"File {file} has no analog counterpart.")
                        continue
                else:
                    analog_file = None
                try:
                    result_file = next(folder.rglob(fname + "*results.npz"))
                    num_of_files_found += 1
                except StopIteration:
                    print(f"File {file} has no result.npz couterpart.")
                    continue
                if self.with_colabeled:
                    try:
                        colabeled_file = next(folder.rglob(fname + "*_colabeled*.npy"))
                        num_of_files_found += 1
                    except StopIteration:
                        print(f"File {file} has no colabeled.npy couterpart.")
                        continue
                else:
                    colabeled_file = None
                try:
                    _ = next(folder.rglob(f"{str(file.name)[:-4]}*.nc"))
                except StopIteration:  # FOV wasn't already analyzed
                    print(
                        summary_str.format(
                            num=num_of_files_found,
                            fluo=file,
                            analog=analog_file,
                            caiman=result_file,
                            colabeled=colabeled_file,
                        )
                    )
                    fluo_files.append(file)
                    result_files.append(result_file)
                    colabeled_files.append(colabeled_file)
                    analog_files.append(analog_file)

        print(
            "\u301C\u301C\u301C\u301C\u301C\u301C\u301C\u301C\u301C\u301C\u301C\u301C\u301C\u301C\u301C\u301C\u301C"
        )
        return fluo_files, analog_files, result_files, colabeled_files

    def _make_table(
        self, fluo_files, analog_files, result_files, colabeled_files
    ) -> pd.DataFrame:
        """
        Turns list of pathlib.Path objects into a DataFrame.
        """
        columns = ["tif", "caiman", "analog", "colabeled"]
        data_files = pd.DataFrame([], columns=columns)
        to_zip = [fluo_files, result_files, analog_files, colabeled_files]
        files_iter = zip(*to_zip)

        for idx, files_tup in enumerate(files_iter):
            cur_row = pd.DataFrame([files_tup], columns=columns, index=[idx])
            data_files = data_files.append(cur_row)
        return data_files


@attr.s(slots=True, hash=True)
class CalciumAnalysisOverTime:
    """ Analysis class that parses the output of CaImAn "results.npz" files.
    Usage: run the "run_batch_of_timepoints" method, which will go over all FOVs
    that were recorded in this experiment.
    "folder_globs" is a dictionary of folder name and glob strings, which allows
    you to analyze several directories of data, each with its own glob pattern.
    If serialize is True, it will write to disk each FOV's DataArray, as well
    as the concatenated DataArray to make future processing faster.
    If you've already serialized your data, use "generate_ds_per_day" to continue
    the downstream analysis of your files by concatenating all relevant files into
    one large database which can be analyzed with downstream scripts that may be
    found in "calcium_trace_analysis.py".
    """

    files_table = attr.ib(validator=instance_of(pd.DataFrame))
    serialize = attr.ib(default=False, validator=instance_of(bool))
    folder_globs = attr.ib(default={Path("."): "*.tif"}, validator=instance_of(dict))
    analog = attr.ib(
        default=AnalogAcquisitionType.NONE, validator=instance_of(AnalogAcquisitionType)
    )
    regex = attr.ib(default=attr.Factory(dict), validator=instance_of(dict))
    list_of_fovs = attr.ib(init=False)
    concat = attr.ib(init=False)

    def run_batch_of_timepoints(self, results_folder):
        """
        Main method to analyze all FOVs in all timepoints in all experiments.
        Generally used for TAC experiments, which have multiple FOVs per mouse, and
        an experiment design which spans multiple days.
        The script expects a filename containing the following "self.fov_analysis_files.append(fields)"::
            * Mouse ID (digits at the beginning of filename)
            * Either 'HYPER' or 'HYPO'
            * 'DAY_0/1/n'
            * 'FOV_n'
        After creating a xr.Dataset out of each file, the script will write this DataArray to
        disk (only if it doesn't exist yet, and only if self.serialize is True) to make future processing faster.
        Finally, it will take all created DataArrays and concatenate them into a single DataArray,
        that can also be written to disk using the "serialize" attribute.
        The `**regex` kwargs-like parameter is used to manually set the regex
        that will parse the metadata from the file name. The default regexes are
        described above. Valid keys are "id_reg", "fov_reg", "cond_reg" and "day_reg".
        """

        # Multiprocessing doesn't work due to the fact that not all objects are
        # pickleable
        # with mp.Pool() as pool:
        #     self.list_of_fovs = pool.map(
        #         self._mp_process_timepoints, self.files_table.itertuples(index=False)
        #     )
        self.list_of_fovs = []
        for row in self.files_table.itertuples():
            self.list_of_fovs.append(self._mp_process_timepoints(row))
        self.generate_ds_per_day(results_folder)

    def _mp_process_timepoints(self, files_row: Tuple):
        """
        A function for a single process that takes three conjugated files - i.e.
        three files that belong to the same recording, and processes them.
        """
        print(f"Parsing {files_row.tif}")
        fov = self._analyze_single_fov(files_row, analog=self.analog, **self.regex)
        return str(fov.metadata.fname)[:-4] + ".nc"

    def _analyze_single_fov(
        self, files_row, analog=AnalogAcquisitionType.NONE, **regex
    ):
        """ Helper function to go file by file, each with its own fluorescence and
        possibly analog data, and run the single FOV parsing on it """

        meta = FluoMetadata(files_row.tif, num_of_channels=2, **regex)
        meta.get_metadata()
        fov = SingleFovParser(
            analog_fname=files_row.analog,
            results_fname=files_row.caiman,
            metadata=meta,
            analog=analog,
            summarize_in_plot=True,
        )
        fov.parse()
        plt.close()
        if self.serialize:
            fov.add_metadata_and_serialize()
        return fov

    def generate_ds_per_day(self, results_folder: Path, globstr="*FOV*.nc", recursive=True):
        """
        Parse .nc files that were generated from the previous analysis
        and chain all "DAY_X" Datasets together into a single list.
        This list is then concatenated in to a single DataArray, creating a
        large data structure for each experimental day.
        If we arrived here from "run_batch_of_timepoints()", the data is already
        present in self.list_of_fovs. Otherwise, we have to manually find the
        files using a default glob string that runs on each folder in
        self.folder_globs.
        Saves all day-data into self.results_folder.
        """
        fovs_by_day = defaultdict(list)
        try:  # coming from run_batch_of_timepoints()
            all_files = self.list_of_fovs
        except AttributeError:
            if recursive:
                all_files = [folder.rglob(globstr) for folder in self.folder_globs]
            else:
                all_files = [folder.glob(globstr) for folder in self.folder_globs]
            all_files = itertools.chain(*all_files)

        for file in all_files:
            if "NEW_crystal_skull_TAC_161018" in str(file) or "crystal_skull_TAC_180719" in str(file) or "602_HYPER_HYPO_DAY_0_AND_ALL" in str(file):
                continue
            print(file)
            try:
                day = int(xr.open_dataset(file).day)
            except AttributeError:  # older datasets
                continue
            fovs_by_day[day].append(file)

        self._concat_fovs(fovs_by_day, results_folder)

    def _concat_fovs(self, fovs_by_day: dict, results_folder: Path):
        """
        Take the list of FOVs and turn them into a single DataArray. Lastly it will
        write this DataArray to disk.
        fovs_by_day: Dictionary with its keys being the days of experiment (0, 1, ...) and
        values as a list of filenames.
        """
        print("Concatenating all FOVs...")
        fname_to_save = "data_of_day_"
        for day, file_list in fovs_by_day.items():
            try:
                file = next(results_folder.glob(fname_to_save + str(day) + ".nc"))
                print(f"Found {str(file)}, not concatenating")
            except StopIteration:  # .nc file doesn't exist
                print(f"Concatenating day {day}")
                data_per_day = []
                for file in file_list:
                    try:
                        data_per_day.append(xr.open_dataset(file).load())
                    except FileNotFoundError:
                        pass
                concat = xr.concat(data_per_day, dim="fname")
                # Fix datatype conversion of epoch_times to floats:
                asbool = concat["epoch_times"].data.astype(np.bool)
                nans = np.where(np.isnan(concat["epoch_times"].data))
                asbool[nans] = False
                concat["epoch_times"] = (["fname", "epoch", "time"], asbool)
                self.concat = concat
                concat.to_netcdf(
                    str(results_folder / f"{fname_to_save + str(day)}.nc"),
                    mode="w",
                )

    def _get_metadata(self, list_of_da: list, key: str, default):
        """ Finds ands returns metadata from existing DataArrays """
        val = default
        for da in list_of_da:
            try:
                val = da.attrs[key]
            except KeyError:
                continue
            else:
                break
        return val


if __name__ == "__main__":
    home = Path("/data/David/")
    folder = Path("TAC_group_7_baseline")
    results_folder = home / folder
    assert results_folder.exists()
    globstr = "*.tif"
    folder_and_files = {home / folder: globstr}
    analog_type = AnalogAcquisitionType.TREADMILL
    filefinder = FileFinder(
        results_folder=results_folder,
        folder_globs=folder_and_files,
        analog=analog_type,
        with_colabeled=False,
    )
    files_table = filefinder.find_files()
    regex = {
        "cond_reg": r"condition_(\w+)_mag",
        "id_reg": r"mouse_(\d+)_day",
        "fov_reg": r"fov_(\d+)_condition",
        "day_reg": r"day_(\d)_fov"
    }
    res = CalciumAnalysisOverTime(
        files_table=files_table,
        serialize=True,
        folder_globs=folder_and_files,
        analog=analog_type,
        regex=regex,
    )
    res.run_batch_of_timepoints(results_folder)
    # res.generate_ds_per_day(results_folder, '*.nc', recursive=True)

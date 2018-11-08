"""
__author__ = Hagai Hargil
"""
import attr
from enum import Enum
import tifffile
from scipy.ndimage.morphology import binary_fill_holes
from attr.validators import instance_of
from pathlib import Path
import pandas as pd
import os
import re
from collections import defaultdict
import itertools
import numpy as np
from datetime import datetime
import multiprocessing as mp
import xarray as xr
import matplotlib.pyplot as plt

from fluo_metadata import FluoMetadata
from analog_trace import AnalogTraceAnalyzer
from trace_converter import RawTraceConverter, ConversionMethod
import caiman_funcs_for_comparison
from single_fov_analysis import SingleFovParser
from calcium_trace_analysis import Condition


class Epoch(Enum):
    """
    All possible TAC epoch combinations
    """
    ALL = 'all'
    RUN = 'run'
    STAND = 'stand'
    STIM = 'stim'
    JUXTA = 'juxta'
    SPONT = 'spont'
    RUN_STIM = 'run_stim'
    RUN_JUXTA = 'run_juxta'
    RUN_SPONT = 'run_spont'
    STAND_STIM = 'stand_stim'
    STAND_JUXTA = 'stand_juxta'
    STAND_SPONT = 'stand_spont'


@attr.s(slots=True)
class CalciumAnalysisOverTime:
    """ A replacement\refactoring for AnalyzeCalciumOverTime.
    Usage: run the "run_batch_of_timepoints" method, which will go over all FOVs
    that were recorded in this experiment.
    The "file_glob" variable is a list of glob strings in that folder. The simplest
    case is ['*.tif'], but it allows you to parse files from different folders inside
    one main folder by specifying the folder name in the glob string.
    If serialize is True, it will write to disk each FOV's DataArray, as well
    as the concatenated DataArray to make future processing faster.
    If you've already serialized your data, use "load_batch_of_timepoints" to continue
    the downstream analysis of your files.
    """
    results_folder = attr.ib(validator=instance_of(Path))
    folder_globs = attr.ib(default={Path('.'): '*.tif'}, validator=instance_of(dict))
    serialize = attr.ib(default=False, validator=instance_of(bool))
    fluo_files = attr.ib(init=False)
    result_files = attr.ib(init=False)
    analog_files = attr.ib(init=False)
    sliced_fluo = attr.ib(init=False)
    list_of_fovs = attr.ib(init=False)

    def _find_all_relevant_files(self):
        self.fluo_files = []
        self.analog_files = []
        self.result_files = []
        for folder, globstr in self.folder_globs.items():
            for file in folder.rglob(globstr):
                if 'CHANNEL' in str(file):
                    pass
                try:
                    analog_file = next(folder.rglob(f'{str(file.name)[:-4]}*analog.txt'))
                except StopIteration:
                    print(f"File {file} has no analog counterpart.")
                    continue
                try:
                    result_file = next(folder.rglob(f'{str(file.name)[:-4]}*results.npz'))
                except StopIteration:
                    print(f"File {file} has no result.npz couterpart.")
                    continue
                try:
                    fov_analysis_file = next(folder.rglob(f'{str(file.name)[:-4]}*.nc'))
                except StopIteration:  # FOV wasn't already analyzed
                    print(f"Found triplet of files:\nfluo: {file},\nanalog: {analog_file}\nresults: {result_file}")
                    self.fluo_files.append(file)
                    self.analog_files.append(analog_file)
                    self.result_files.append(result_file)

        print("\u301C\u301C\u301C\u301C\u301C\u301C\u301C\u301C\u301C\u301C\u301C\u301C\u301C\u301C\u301C\u301C\u301C")

    def run_batch_of_timepoints(self, **regex):
        """
        Main method to analyze all FOVs in all timepoints in all experiments. 
        Generally used for TAC experiments, which have multiple FOVs per mouse, and 
        an experiment design which spans multiple days.
        The script expects a filename containing the following "fself.fov_analysis_files.append(None)ields":
            Mouse ID (digits at the beginning of filename)
            Either 'HYPER' or 'HYPO'
            'DAY_0/1/n'
            'FOV_n'
        After creating a xr.DataArray out of each file, the script will write this DataArray to
        disk (only if it doesn't exist yet, and only if self.serialize is True) to make future processing faster.
        Finally, it will take all created DataArrays and concatenate them into a single DataArray, 
        that can also be written to disk using the "serialize" attribute.
        The `**regex` kwargs-like parameter is used to manually set the regex
        that will parse the metadata from the file name. The default regexes are 
        described above. Valid keys are "id_reg", "fov_reg" and "day_reg".
        """
        self.list_of_fovs = []
        self._find_all_relevant_files()
        assert len(self.fluo_files) == len(self.analog_files) == len(self.result_files)

        for file_fluo, file_result, file_analog in zip(self.fluo_files, self.result_files, self.analog_files):
            print(f"Parsing {file_fluo}")
            fov = self._analyze_single_fov(file_fluo, file_result, file_analog, **regex)
            self.list_of_fovs.append(str(fov.metadata.fname)[:-4] + ".nc")
        self.generate_da_per_day()

    def _analyze_single_fov(self, fname_fluo, fname_results, fname_analog, **regex):
        """ Helper function to go file by file, each with its fluorescence and analog data,
        and run the single FOV parsing on it """

        meta = FluoMetadata(fname_fluo, **regex)
        meta.get_metadata()
        fov = SingleFovParser(analog_fname=fname_analog, fluo_fname=fname_results,
                              metadata=meta)
        fov.parse()
        if self.serialize:
            fov.add_metadata_and_serialize()
        return fov

    def generate_da_per_day(self, globstr='*FOV*.nc'):
        """ 
        Parse .nc files that were generated from the previous analysis
        and chain all "DAY_X" DataArrays together into a single list. 
        This list is then concatenated in to a single DataARray, creating a 
        large data structure for each experimental day.
        If we arrived here from "run_batch_of_timepoints()", the data is already
        present in self.list_of_fovs. Otherwise, we have to manually find the 
        files using a default glob string that runs on each folder in
        self.folder_globs.
        Saves all day-data into self.results_folder.
        """
        fovs_by_day = defaultdict(list)
        day_reg = re.compile(r'_DAY_*(\d+)_')
        try:  # coming from run_batch_of_timepoints()
            all_files = self.list_of_fovs
        except AttributeError:
            all_files = [folder.rglob(globstr) for folder in self.folder_globs]
            all_files = itertools.chain(*all_files)

        for file in all_files:
            print(file)
            try:
                day = int(day_reg.findall(str(file))[0])
            except IndexError:
                day = 999
            fovs_by_day[day].append(file)

        self._concat_fovs(fovs_by_day)

    def _concat_fovs(self, fovs_by_day: dict):
        """
        Take the list of FOVs and turn them into a single DataArray. Lastly it will
        write this DataArray to disk.
        fovs_by_day: Dictionary with its keys being the days of experiment (0, 1, ...) and
        values as a list of filenames.
        """
        print("Concatenating all FOVs...")
        fname_to_save = 'data_of_day_'
        for day, file_list in fovs_by_day.items():
            try:
                next(self.results_folder.glob(fname_to_save + str(day) + '.nc'))
            except StopIteration:   #.nc file doesn't exist
                print(f"Concatenating day {day}")
                data_per_day = []
                for file in file_list:
                    try:
                        data_per_day.append(xr.open_dataarray(file).load())
                    except FileNotFoundError:
                        pass
                concat = xr.concat(data_per_day, dim='neuron')
                concat.attrs['fps'] = self._get_metadata(data_per_day, 'fps', 30)
                concat.attrs['stim_window'] = self._get_metadata(data_per_day, 'stim_window', 1.5)
                concat.attrs['day'] = day
                concat.to_netcdf(str(self.results_folder / f"{fname_to_save + str(day)}.nc"), mode='w',
                                format='NETCDF3_64BIT')

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


if __name__ == '__main__':
    results_folder = Path(r'/data/David/TAC_together_nov18')
    assert results_folder.exists()
    # folder_and_files = {Path('/data/David/NEW_crystal_skull_TAC_161018'): 'DAY*/*/*.tif',
    #                     Path('/data/David/crystal_skull_TAC_180719'): '626*/*.tif'}
    folder_and_files = {Path('/data/David/NEW_crystal_skull_TAC_161018'): 'DAY*/*/*.tif'}
    res = CalciumAnalysisOverTime(results_folder=results_folder, serialize=True, 
                                  folder_globs=folder_and_files)
    # regex = {'id_reg': r'_(\d+?)_X10',
    #          'cond_reg': r'^([a-zA-Z]+?)_[0-9]'}
    # res.run_batch_of_timepoints()
    res.generate_da_per_day()

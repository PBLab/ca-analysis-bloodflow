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
import numpy as np
from datetime import datetime
# from os.path import splitext
import multiprocessing as mp
import xarray as xr
import matplotlib.pyplot as plt

from fluo_metadata import FluoMetadata
from analog_trace import AnalogTraceAnalyzer
from trace_converter import RawTraceConverter, ConversionMethod
import caiman_funcs_for_comparison
from single_fov_analysis import SingleFovParser
from calcium_trace_analysis import CalciumAnalyzer, Condition


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
    If serialize is True, it will write to disk each FOV's DataArray, to make future
    processing faster.
    If you've already serialized your data, use "load_batch_of_timepoints" to continue
    the downstream analysis of your files.
    """
    foldername = attr.ib(validator=instance_of(Path))
    file_glob = attr.ib(default='*.tif', validator=instance_of(str))
    serialize = attr.ib(default=False, validator=instance_of(bool))
    fluo_files = attr.ib(init=False)
    result_files = attr.ib(init=False)
    analog_files = attr.ib(init=False)
    sliced_fluo = attr.ib(init=False)
        
    def _find_all_relevant_files(self):
        self.fluo_files = []
        self.analog_files = []
        self.result_files = []
        for file in self.foldername.rglob(self.file_glob):
            if 'CHANNEL' in str(file):
                pass
            try:
                analog_file = next(self.foldername.rglob(f'{str(file.name)[:-4]}*analog.txt'))
            except StopIteration:
                print(f"File {file} has no analog counterpart.")
                continue
            try:
                result_file = next(self.foldername.rglob(f'{str(file.name)[:-4]}*CHANNEL*results.npz'))
            except StopIteration:
                print(f"File {file} has no result.npz couterpart.")
                continue
            print(f"Found triplet of files:\nfluo: {file},\nanalog: {analog_file}\nresults: {result_file}")
            self.fluo_files.append(file)
            self.analog_files.append(analog_file)
            self.result_files.append(result_file)

        print("\u301C\u301C\u301C\u301C\u301C\u301C\u301C\u301C\u301C\u301C\u301C\u301C\u301C\u301C\u301C\u301C\u301C")

    def run_batch_of_timepoints(self):
        """
        Main method to analyze all FOVs in all timepoints in all experiments. 
        Generally used for TAC experiments, which have multiple FOVs per mouse, and 
        an experiment design which spans multiple days.
        The script expects a filename containing the following "fields":
            Mouse ID (digits at the beginning of filename)
            Either 'HYPER' or 'HYPO'
            'DAY_0/1/n'
            'FOV_n'
        After creating a xr.DataArray out of each file, the script will write this DataArray to
        disk (only if it doesn't exist yet, and only if self.serialize is True) to make future processing faster.
        """
        self._find_all_relevant_files()
        assert len(self.fluo_files) == len(self.analog_files) == len(self.result_files)

        for file_fluo, file_result, file_analog in zip(self.fluo_files, self.result_files, self.analog_files):
            print(f"Parsing {file_fluo}")
            self.list_of_fovs.append(self._analyze_single_fov(file_fluo, file_result, file_analog))

    def _analyze_single_fov(self, fname_fluo, fname_results, fname_analog):
        """ Helper function to go file by file, each with its fluorescence and analog data,
        and run the single FOV parsing on it """

        meta = FluoMetadata(fname_fluo)
        meta.get_metadata()
        fov = SingleFovParser(analog_fname=fname_analog, fluo_fname=fname_results,
                              metadata=meta)
        fov.parse()
        if self.serialize:
            fov.add_metadata_and_serialize()
        return fov

    def load_batch_of_timepoints(self):
        """ 
        Find all .nc files that were generated from the previous analysis
        and chain them together into a single list. This list is then concatenated
        into a single xr.DataArray. The new coordinates order is 
        (epoch, neuron, time, mouse_id, fov, condition, day).
        """
        print("Found the following NetCDF files:")
        all_nc = self.foldername.rglob('*.nc')
        list_of_fovs = []
        for file in all_nc:
            list_of_fovs.append(xr.open_dataarray(file))
            print(file.name)
        self.sliced_fluo = xr.concat(list_of_fovs, dim='neuron')
        self.sliced_fluo.attrs['fps'] = list_of_fovs[0].attrs['fps']
        self.sliced_fluo.attrs['stim_window'] = list_of_fovs[0].attrs['stim_window']
        self.sliced_fluo.to_netcdf(str(self.foldername / Path("all_fovs_dataset.nc")),
                                   mode='w', format='NETCDF3_64BIT')


if __name__ == '__main__':
    folder = Path.home() / Path(r'data/David/crystal_skull_TAC_180719')
    assert folder.exists()
    res = CalciumAnalysisOverTime(foldername=folder, serialize=True)
    # res.run_batch_of_timepoints()
    res.load_batch_of_timepoints()
    plt.show(block=False)

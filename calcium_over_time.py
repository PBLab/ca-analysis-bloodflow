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
    """
    foldername = attr.ib(validator=instance_of(Path))
    file_glob = attr.ib(default='*.tif', validator=instance_of(str))
    fluo_files = attr.ib(init=False)
    result_files = attr.ib(init=False)
    analog_files = attr.ib(init=False)
    list_of_fovs = attr.ib(init=False)
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
        """
        self.list_of_fovs = []
        self._find_all_relevant_files()
        assert len(self.fluo_files) == len(self.analog_files) == len(self.result_files)

        for file_fluo, file_result, file_analog in zip(self.fluo_files[:], self.result_files[:], self.analog_files[:]):
            print(f"Parsing {file_fluo}")
            self.list_of_fovs.append(self._analyze_single_fov(file_fluo, file_result, file_analog))
        print("Finished processing all files, starting the concatenation...")
        self.sliced_fluo = self.__concat_dataarrays()

    def _analyze_single_fov(self, fname_fluo, fname_results, fname_analog):
        """ Helper function to go file by file, each with its fluorescence and analog data,
        and run the single FOV parsing on it """

        meta = FluoMetadata(fname_fluo)
        meta.get_metadata()
        fov = SingleFovParser(analog_fname=fname_analog, fluo_fname=fname_results,
                              metadata=meta)
        fov.parse()
        return fov

    def __concat_dataarrays(self) -> xr.Dataset:
        """ 
        Parses all exisiting DataArrays, each corresponding to a different FOV, and concatenates
        them into a single xr.Dataset that can later be sliced properly. The method first creates
        a new DataArray object with all metadata as data coordinates, and then tries to concat
        these objects.
        The new coordinates order is (epoch, neuron, time, mouse_id, fov, condition).
        The "day" coordinate is considered the "special" dimension of the final xr.Dataset.
        """
        days_with_data = {}
        for fov in self.list_of_fovs:
            days_with_data[fov.metadata.day] = None

        for fov in self.list_of_fovs:
            raw_data = fov.fluo_analyzed.data
            raw_data = raw_data[..., np.newaxis, np.newaxis, np.newaxis]
            assert len(raw_data.shape) == 6
            coords = {}
            coords['epoch'] = fov.fluo_analyzed['epoch'].values
            coords['neuron'] = fov.fluo_analyzed['neuron'].values
            coords['time'] = fov.fluo_analyzed['time'].values
            coords['mouse_id'] = np.array([fov.metadata.mouse_id])
            coords['fov'] = np.array([fov.metadata.fov])
            coords['condition'] = np.array([fov.metadata.condition])
            darr = xr.DataArray(raw_data, coords=coords, dims=coords.keys())
            day = fov.metadata.day
            try:
                days_with_data[day] = xr.concat([days_with_data[day], darr], dim='neuron')
            except TypeError:  # empty day, need to populate the first one
                days_with_data[day] = darr
        
        ds = xr.Dataset(days_with_data)
        ds.attrs['fps'] = fov.fluo_analyzed.attrs['fps']
        ds.attrs['stim_window'] = fov.fluo_analyzed.attrs['stim_window']
        return ds


if __name__ == '__main__':
    # base_folder = r'/data/David/THY_1_GCaMP_BEFOREAFTER_TAC_290517/'
    # new_folders = [
    #                '747_HYPER_DAY_1__EXP_STIM',
    #                '747_HYPER_DAY_7__EXP_STIM',
    #                '747_HYPER_DAY_14__EXP_STIM']
    # for folder in new_folders:
    #     result = AnalyzeCalciumOverTime(Path(base_folder + folder)).run_batch_of_timepoint()
    # res = AnalyzeCalciumOverTime(Path(r'/data/David/THY_1_GCaMP_BEFOREAFTER_TAC_290517'))\
    #     .read_dataarrays_over_time(epoch=Epoch.ALL)
    # plt.show(block=False)
    folder = Path(r'X:/David/crystal_skull_TAC_180719')
    res = CalciumAnalysisOverTime(foldername=folder)
    res.run_batch_of_timepoints()
    plt.show(block=False)

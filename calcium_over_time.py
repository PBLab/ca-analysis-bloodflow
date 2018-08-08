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
    # analyzed_sliced_hyper = attr.ib(init=False)
    # analyzed_sliced_hypo = attr.ib(init=False)
    
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

        # for file_fluo, file_result, file_analog in zip(self.fluo_files, self.result_files, self.analog_files):
        #     print(f"Parsing {file_fluo}")
        #     self.list_of_fovs.append(self._analyze_single_fov(file_fluo, file_result, file_analog))
        self.list_of_fovs = mp.Pool().starmap(self._analyze_single_fov,
                                              zip(self.fluo_files, self.result_files, self.analog_files))
        print("Finished processing all files, starting the concatenation...")
        # sliced_fluo = xr.concat([fov.fluo_analyzed for fov in self.list_of_fovs],
        #                         dim='neuron')
        # self.analyzed_sliced_hyper = CalciumAnalyzer(sliced_fluo, cond=Condition.HYPER)
        # self.analyzed_sliced_hyper.run_analysis()
        # self.analyzed_sliced_hypo = CalciumAnalyzer(sliced_fluo, cond=Condition.HYPO)
        # self.analyzed_sliced_hypo.run_analysis()

    def _analyze_single_fov(self, fname_fluo, fname_results, fname_analog):
        """ Helper function to go file by file, each with its fluorescence and analog data,
        and run the single FOV parsing on it """

        meta = FluoMetadata(fname_fluo)
        meta.get_metadata()
        fov = SingleFovParser(analog_fname=fname_analog, fluo_fname=fname_results,
                              metadata=meta)
        fov.parse()
        return fov

    def __save(self, data: xr.DataArray, foldername: Path, name: str):
            
        print(f"Saving {name} data as NetCDF...")
        try:
            data.to_netcdf(str(foldername) + '/{name}_DataArray.nc')
        except AttributeError:  # NoneType
            pass


@attr.s(slots=True)
class AnalyzeCalciumOverTime:
    """ OLD """
    foldername = attr.ib(validator=instance_of(Path))
    hyper_glob = attr.ib(default=r'*_HYPER_DAY_*[0-9]__EXP_STIM_*FOV_[0-9]_0000[0-9].tif')
    hypo_glob = attr.ib(default=r'*_HYPO_DAY_*[0-9]__EXP_STIM_*FOV_[0-9]_0000[0-9].tif')
    fps = attr.ib(init=False)
    colors = attr.ib(init=False)
    num_of_channels = attr.ib(init=False)
    start_time = attr.ib(init=False)
    timestamps = attr.ib(init=False)

    def __attrs_post_init__(self):
        self.colors = [f"C{idx}" for idx in range(10)] * 3  # use default matplotlib colormap

    def __find_files(self, glob: str, name: str):
        """ 
        Detect all files in a folder recursively according to glob.
        name lets the function know what is it looking for - HYPER\HYPO.
        Returns a generator with all files found, and the name of
        the last file in that list. 
        """
        files = self.foldername.glob(glob)
        print(f"Found the following {name} files:")
        for file in files:
            print(file)
        return self.foldername.glob(glob), file

    def run_batch_of_timepoint(self):
        """
        Pool all neurons from all FOVs of a single timepoint together and analyze them
        :return:
        """
        all_files_hyper, file = self.__find_files(self.hyper_glob, 'Hyper')
        all_files_hypo, _ = self.__find_files(self.hypo_glob, 'Hypo')

        # Get params
        try:
            with tifffile.TiffFile(str(file)) as f:
                si_meta = f.scanimage_metadata
                self.fps = si_meta['FrameData']['SI.hRoiManager.scanFrameRate']
                self.num_of_channels = len(si_meta['FrameData']['SI.hChannels.channelsActive'])
                self.start_time = str(datetime.fromtimestamp(os.path.getmtime(str(file))))
                self.timestamps = np.arange(len(f.pages)//self.num_of_channels)
        except TypeError:
            self.fps = 15.24

        sliced_fluo_hyper, hyper_data = self.__run_analysis_batch(all_files_hyper, cond='hyper')
        sliced_fluo_hypo, hypo_data = self.__run_analysis_batch(all_files_hypo, cond='hypo')



    def __run_analysis_batch(self, files, cond: str):
        """
        For data over several timepoints after analysis with CaImAn - run the analog analysis and the summation of
        neural data for all timepoints.
        :param files:
        :param cond: 'HYPER', 'HYPO'
        :return:
        """
        all_data = []
        all_analog = []
        for file in files:
            print(f'\nRunning {file}')
            img_neuron, time_vec, fluo_trace, rois = determine_manual_or_auto(filename=file,
                                                                              fps=self.fps,
                                                                              num_of_rois=1,
                                                                              colors=self.colors,
                                                                              num_of_channels=1,
                                                                              channel_to_keep=1)
            all_data.append(fluo_trace)
            analog_data_fname = next(file.parent.glob(f'{str(file.name)[:-4]}*analog.txt'))
            analog_data = pd.read_table(analog_data_fname, header=None,
                                        names=['stimulus', 'run'], index_col=False)
            an_trace = AnalogTraceAnalyzer(str(file), analog_data, framerate=self.fps,
                                           num_of_channels=self.num_of_channels,
                                           start_time=self.start_time,
                                           timestamps=self.timestamps)
            an_trace.run()
            all_analog.append(an_trace * fluo_trace)  # Overloaded __mul__

        # Further analysis of sliced calcium traces follows
        if len(all_analog) > 0:
            sliced_fluo = xr.concat((all_analog), dim='neuron')
            analyzed_data = CalciumAnalyzer(sliced_fluo, cond=cond, plot=False)
            analyzed_data.run_analysis()
            return sliced_fluo, analyzed_data
        else:
            return None, None

    def read_dataarrays_over_time(self, epoch: Epoch):
        """
        Read and parse DataArrays saved as .nc files and display their data
        :return:
        """
        days = ['DAY_0', 'DAY_1', 'DAY_7', 'DAY_14', 'DAY_21']
        hypo, hyper = self.__gen_dict_with_epoch_data(epoch=epoch)
        offsets_hypo = np.arange(len(hypo)*2, step=2)
        offsets_hyper = np.arange(len(hyper)*2, step=2)
        if len(hypo) > len(hyper):
            tick_labels = list(hypo.keys())
            ticks = offsets_hypo
        else:
            tick_labels = list(hyper.keys())
            ticks = offsets_hyper

        data_hypo = [hypo[day][np.isfinite(hypo[day])] for day in hypo.keys()]
        data_hyper = [hyper[day][np.isfinite(hyper[day])] for day in hyper.keys()]

        fig, ax = plt.subplots()
        try:
            ax.violinplot(data_hypo, positions=offsets_hypo+0.25, points=100, showmeans=True)
        except NameError:
            pass
        try:
            parts = ax.violinplot(data_hyper, positions=offsets_hyper-0.25, points=100, showmeans=True)
            [pc.set_facecolor('orange') for pc in parts['bodies']]
        except NameError:
            pass

        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels)
        plt.legend(['Hypo (blue)', 'Hyper (orange)'])
        ax.set_title(f'Hypo-Hyper average dF/F in {epoch.value}')
        ax.set_ylabel('Mean dF/F')
        plt.savefig(f'TAC_{epoch.value}.pdf', transparent=True)

        return hypo, hyper

    def __gen_dict_with_epoch_data(self, epoch: Epoch):

        all_files = self.foldername.rglob(r'*_DataArray.nc')
        days_hyper = {}
        days_hypo = days_hyper.copy()
        for file in all_files:
            reg = re.compile(r'(DAY_\d+)_')
            cur_day = reg.findall(str(file))[0]
            da = xr.open_dataarray(file)
            if epoch is not Epoch.ALL:
                if 'HYPER' in str(file):
                    days_hyper[cur_day] = np.nanmean(da.loc[epoch.value].values, axis=1)
                elif 'HYPO' in str(file):
                    days_hypo[cur_day] = np.nanmean(da.loc[epoch.value].values, axis=1)
            else:
                if 'HYPER' in str(file):
                    days_hyper[cur_day] = np.nanmean(da.values, axis=1)
                elif 'HYPO' in str(file):
                    days_hypo[cur_day] = np.nanmean(da.values, axis=1)

        return days_hypo, days_hyper

    def calc_df_f_over_time(self, filename):
        fig, ax = self.__calculate_mean_dff_after_caiman_segmentation(filename)

    def __calculate_mean_dff_after_caiman_segmentation(self, tif_filename):
        """
        Since the dF/F calculation of CaImAn isn't trustworthy, we calcluate the dF/F values ourselves
        according to the ROIs that CaImAn found.
        :return:
        """
        name = os.splitext(tif_filename.name)[0]
        parent_folder = tif_filename.parent
        try:
            corresponding_npz = next(parent_folder.glob(name + "*results.npz"))
        except StopIteration:
            raise UserWarning(f"File {tif_filename} doesn't have a corresponding .npz file.")

        neurons = self.__gen_masked_image(corresponding_npz, tif_filename)
        df_f_mat = RawTraceConverter(conversion_method=ConversionMethod.DFF,
                                        raw_data=neurons).convert()
        time_vec = np.arange(len(df_f_mat))
        time_vec = np.tile(time_vec, (1, df_f_mat.shape[1]))

        fig, ax = plt.subplots()
        ax.plot(time_vec, df_f_mat)
        return fig, ax

    def __gen_masked_image(self, file, tif_filename) -> np.ndarray:
        """
        With the list of coordinates create a masked image, labeled with integers
        :param file:
        :return:
        """
        data = tifffile.imread(str(tif_filename))
        dims = data.shape[1:]
        data_crd = np.load(str(file), encoding='bytes')['crd_good']
        neurons = np.zeros((data.shape[0], len(data_crd)))  # frames x num_of_neurons

        for idx, cell in enumerate(data_crd):
            mask = np.zeros(dims)
            mask[cell[b'coordinates'][1:-1, 0], cell[b'coordinates'][1:-1, 1]] = 1
            mask = binary_fill_holes(mask)
            neurons[:, idx] = ((data * mask).mean(axis=-1))

        return neurons

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
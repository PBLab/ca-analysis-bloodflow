import pathlib
import re
from typing import Tuple, Any, Union
from itertools import product
import os
from datetime import datetime
from collections import namedtuple
from enum import Enum

import tifffile
import pandas as pd
import numpy as np
import attr
from attr.validators import instance_of
import xarray as xr

# Constant values for the analog acquisiton
TYPICAL_JUXTA_VALUE = -480
TYPICAL_PUFF_VALUE = -27180


class AnalogAcquisitionType(Enum):
    NONE = "NONE"
    OLD = "OLD"
    MRDUINO = "MRDUINO"
    TREADMILL = "TREADMILL"
    TREADROWS = "TREADROWS"


def analog_trace_runner(
    tif_filename,
    analog_trace,
    analog_type,
    framerate,
    start_time,
    timestamps,
    occluder,
    occ_metadata=None,
):
    if analog_type is AnalogAcquisitionType.OLD:
        analysis = AnalogAnalysisOld(
            tif_filename,
            analog_trace,
            framerate,
            start_time,
            timestamps,
            occluder=occluder,
            occ_metadata=occ_metadata,
        )
    elif analog_type is AnalogAcquisitionType.MRDUINO:
        analysis = AnalogAnalysisMrduino(
            tif_filename,
            analog_trace,
            framerate,
            start_time,
            timestamps,
            occluder=occluder,
            occ_metadata=occ_metadata,
        )
    elif analog_type is AnalogAcquisitionType.TREADROWS:
        analysis = AnalogAnalysisTreadmillRows(
            tif_filename,
            analog_trace,
            framerate,
            start_time,
            timestamps,
            occluder=occluder,
            occ_metadata=occ_metadata,
        )
    elif analog_type is AnalogAcquisitionType.TREADMILL:
        analysis = AnalogAnalysisTreadmill(
            tif_filename,
            analog_trace,
            framerate,
            start_time,
            timestamps,
            occluder=occluder,
            occ_metadata=occ_metadata,
        )
    else:
        raise TypeError("Invalid analog acquisition type")
    analysis.run()
    return analysis


@attr.s
class AnalyzedAnalogTrace:
    """
    Fit and match the analog trace corresponding to the puffs and running of the mouse with the neuronal trace.

    analog_trace is a DataFrame with its first column being the air puff data,
    and second being the run vector data.
    Usage:
        AnalogTraceAnalyzer(tif_filename, analog_trace).run()
    # TODO: ADD SUPPORT FOR COLABELING INDICES
    """

    tif_filename = attr.ib(
        validator=instance_of(pathlib.Path)
    )  # Timelapse (doesn't need to be separated)
    analog_trace = attr.ib(
        validator=instance_of(pd.DataFrame)
    )  # .txt file from ScanImage
    framerate = attr.ib(validator=instance_of(float))
    start_time = attr.ib(validator=instance_of(str))
    timestamps = attr.ib(validator=instance_of(np.ndarray))
    puff_length = attr.ib(default=1.0, validator=instance_of(float))  # sec
    response_window = attr.ib(default=0.5, validator=instance_of(float))  # sec
    buffer_after_stim = attr.ib(default=1.0, validator=instance_of(float))  # sec
    move_thresh = attr.ib(default=0.25, validator=instance_of(float))  # V
    sample_rate = attr.ib(default=1000, validator=instance_of(int))  # Hz
    occluder = attr.ib(default=False, validator=instance_of(bool))
    occ_metadata = attr.ib(default=None)  # namedtuple of the occlusion frame durations
    stim_vec = attr.ib(init=False)
    juxta_vec = attr.ib(init=False)
    spont_vec = attr.ib(init=False)
    run_vec = attr.ib(init=False)
    stand_vec = attr.ib(init=False)
    occluder_vec = attr.ib(init=False)
    before_occ_vec = attr.ib(init=False)
    after_occ_vec = attr.ib(init=False)

    def __attrs_post_init__(self):
        if self.occ_metadata is None:
            occ = namedtuple("Occluder", ("before", "during"))
            self.occ_metadata = occ(100, 200)

    def _find_peaks(self, vec=None) -> Tuple[np.ndarray, np.ndarray]:
        """ Find the starts of the puff events and mark their duration.
        Returns a vector length of which is the same size as the TIF data,
        with 1 wherever the a puff or a juxta puff occurred, and 0 elsewhere.
        If vec is None (default), uses self.analog_trace.stimulus (for
        backwards compatability purposes), else uses the given data.
        """
        if vec is None:
            vec = self.analog_trace.stimulus.to_numpy()
        else:
            vec = vec.to_numpy()
        diffs_all = np.where(np.diff(vec) < -100)[0]
        diffs_true = np.where(np.diff(vec) < -1000)[0]
        intersect = np.in1d(diffs_all, diffs_true)
        true_puff_idx = diffs_all[intersect]
        juxta_puff_idx = diffs_all[~intersect]
        true_puff_times = np.zeros_like(vec)
        juxta_puff_times = np.zeros_like(vec)
        if len(true_puff_idx) > 0:
            true_puff_times = self._iter_over_puff_times(true_puff_idx, vec.shape)
        if len(juxta_puff_idx) > 0:
            juxta_puff_times = self._iter_over_puff_times(juxta_puff_idx, vec.shape)

        return true_puff_times.astype(np.float64), juxta_puff_times.astype(np.float64)

    def _iter_over_puff_times(self, puff_idx, vec_len):
        max_puff_length = int(self.framerate * self.puff_length)
        buffer_after_stim_frames = int(self.framerate * self.buffer_after_stim)
        puff_times = np.zeros(vec_len)
        puff_limits = np.where(np.diff(puff_idx) > max_puff_length)[0]
        start_puff_indices = puff_limits + 1
        start_puff_indices = np.concatenate(([0], start_puff_indices))
        end_puff_indices = np.concatenate((puff_limits, [len(puff_idx) - 1]))
        for start_of_puff, end_of_puff in zip(start_puff_indices, end_puff_indices):
            puff_times[
                puff_idx[start_of_puff] : (
                    puff_idx[end_of_puff] + buffer_after_stim_frames
                )
            ] = 1

        return puff_times

    def _populate_occluder(self):
        self.before_occ_vec = np.full(self.timestamps.shape, np.nan)
        self.occluder_vec = np.full(self.timestamps.shape, np.nan)
        self.after_occ_vec = np.full(self.timestamps.shape, np.nan)

        tot_len_during = self.occ_metadata.before + self.occ_metadata.during
        self.before_occ_vec[: self.occ_metadata.before] = 1
        self.occluder_vec[self.occ_metadata.before : tot_len_during] = 1
        self.after_occ_vec[tot_len_during:] = 1

    def _populate_run(self) -> np.ndarray:
        """
        Wherever the analog voltage passes the threshold, assign a 1 value
        :return: None
        """
        run_vec = np.full(self.analog_trace.shape[0], np.nan)
        run_vec[self.analog_trace.run > self.move_thresh] = 1
        return run_vec

    def _populate_spont(
        self, stim_vec: np.ndarray, juxta_vec: np.ndarray
    ) -> np.ndarray:
        """
        Wherever the juxta and stim vectors are zero - write 1, else write nan.
        :return: None
        """
        all_stims = np.nan_to_num(stim_vec) + np.nan_to_num(juxta_vec)
        spont_vec = np.logical_not(all_stims)
        spont_vec = np.where(spont_vec, 1, np.nan)
        return spont_vec

    def _convert_to_series(self):
        self.stim_vec = pd.Series(self.stim_vec)
        self.juxta_vec = pd.Series(self.juxta_vec)
        self.run_vec = pd.Series(self.run_vec)
        self.spont_vec = pd.Series(self.spont_vec)
        self.stand_vec = pd.Series(self.stand_vec)
        if self.occluder:
            self.before_occ_vec = pd.Series(self.before_occ_vec)
            self.occluder_vec = pd.Series(self.occluder_vec)
            self.after_occ_vec = pd.Series(self.after_occ_vec)

    def _init_vecs(self):
        self.stim_vec = np.full(self.timestamps.shape, np.nan)
        self.juxta_vec = np.full(self.timestamps.shape, np.nan)
        self.run_vec = np.full(self.timestamps.shape, np.nan)
        self.spont_vec = np.full(self.timestamps.shape, np.nan)
        self.stand_vec = np.full(self.timestamps.shape, np.nan)

    def _fit_frames_to_analog(
        self,
        stim_vec: np.ndarray,
        juxta_vec: np.ndarray,
        run_vec: np.ndarray,
        spont_vec: np.ndarray,
    ):
        samples_per_frame = int(np.ceil(self.sample_rate / self.framerate))
        starting_idx = np.linspace(
            0, len(stim_vec), num=len(self.timestamps), dtype=np.int64, endpoint=False
        )
        end_idx = starting_idx + samples_per_frame

        for frame_idx, (start, end) in enumerate(zip(starting_idx, end_idx)):
            self.stim_vec[frame_idx] = (
                1.0 if np.nanmean(stim_vec[start:end]) > 0.5 else np.nan
            )
            self.juxta_vec[frame_idx] = (
                1.0 if np.nanmean(juxta_vec[start:end]) > 0.5 else np.nan
            )
            self.run_vec[frame_idx] = (
                1.0 if np.nanmean(run_vec[start:end]) > 0.5 else np.nan
            )
            self.spont_vec[frame_idx] = (
                1.0 if np.nanmean(spont_vec[start:end]) > 0.5 else np.nan
            )

        stand_vec = np.logical_not(np.nan_to_num(self.run_vec))
        self.stand_vec = np.where(stand_vec, 1.0, np.nan)

    def _zero_to_nan(self, vec: np.ndarray) -> np.ndarray:
        """
        Turns zero entries in an array to nans
        """
        vec[vec == 0] = np.nan
        return vec

    def _get_metadata(self) -> Tuple[int, ...]:
        """
        Retrieves metadata from ScanImage files.
        """
        with tifffile.TiffFile(str(self.tif_filename)) as f:
            meta = f.scanimage_metadata
        num_of_lines = int(meta["FrameData"]["SI.hRoiManager.linesPerFrame"])
        num_of_frames = int(meta["FrameData"]["SI.hStackManager.framesPerSlice"])
        return num_of_lines, num_of_frames

    @staticmethod
    def normalize_vec(vec: pd.Series) -> pd.Series:
        vec -= vec.min()
        vec = vec / vec.max()
        return vec

    def _populate_stand(self) -> np.ndarray:
        """
        Create the vector that shows the times at which the mouse
        was standing, i.e. when it wasn't running.
        """
        stand_vec = np.logical_not(np.nan_to_num(self.run_vec))
        return np.where(stand_vec, 1.0, np.nan)

    def __mul__(self, other: np.ndarray) -> xr.DataArray:
        """
        Multiplying an AnalogTrace with a numpy array containing the fluorescent trace results
        in an xarray containing the sliced data. The numpy array will start from zero.
        :param other: np.ndarray
        :return xr.DataArray:
        """
        assert isinstance(other, np.ndarray)

        coords_of_neurons = np.arange(other.shape[0])

        # To find all possible combinations of running and stimulus we run a Cartesian product
        dims = ["epoch", "neuron", "time"]
        movement = ["run", "stand", None]
        puff = ["stim", "juxta", "spont", None]
        ones = np.ones_like(self.timestamps, dtype=np.uint8)
        move_data = [self.run_vec, self.stand_vec, ones]
        puff_data = [self.stim_vec, self.juxta_vec, self.spont_vec, ones]
        coords = [movement, puff]
        analog_data = [move_data, puff_data]
        if self.occluder:
            occluder = ["before_occ", "during_occ", "after_occ", None]
            coords.append(occluder)
            occ_data = [
                self.before_occ_vec,
                self.occluder_vec,
                self.after_occ_vec,
                ones,
            ]
            analog_data.append(occ_data)
        all_coords = []
        all_data = []
        for coord, datum in zip(product(*coords), product(*analog_data)):
            try:
                all_coords.append("_".join((filter(None.__ne__, coord))))
            except IndexError:
                pass
            # Filter "None"s and multiply to find the joint area
            prod = np.array([x for x in datum if type(x) is pd.Series]).prod(axis=0)
            all_data.append(prod)

        all_coords[-1] = "all"  # last item is ''

        da = xr.DataArray(
            np.zeros((len(all_coords), other.shape[0], other.shape[1])),
            coords=[
                ("epoch", all_coords),
                ("neuron", coords_of_neurons),
                ("time", np.arange(other.shape[1]) / self.framerate),
            ],
            dims=dims,
        )  # self.timestamps

        for coor, vec in zip(all_coords, all_data):
            da.loc[coor] = other * np.atleast_2d(vec)

        da.attrs["fps"] = self.framerate
        da.attrs["stim_window"] = self.puff_length + self.buffer_after_stim
        return da


@attr.s
class AnalogAnalysisTreadmill(AnalyzedAnalogTrace):
    """
    Analysis for the second version of the treadmill output,
    where each frame of the stack was given an analog value.
    """

    num_of_lines = attr.ib(init=False)
    num_of_frames = attr.ib(init=False)

    def run(self):
        self.num_of_lines, self.num_of_frames = self._get_metadata()
        stim_vec, juxta_vec = self._find_peaks()
        self.stim_vec = self._zero_to_nan(stim_vec)
        self.juxta_vec = self._zero_to_nan(juxta_vec)
        run_vec = self.normalize_vec(self.analog_trace.run.to_numpy())
        self.run_vec = self._populate_run(run_vec)
        self.spont_vec = self._populate_spont(stim_vec, juxta_vec)
        self.stand_vec = self._populate_stand()
        if self.occluder:
            self._populate_occluder()

        self._convert_to_series()

    def _populate_run(self, run_vec: np.ndarray) -> np.ndarray:
        """
        Processes the running analog data from the encoders.
        Looks for 1 second long periods in which the mouse ran
        by diffing the raw output of the encoder and checking if
        it passes some magic number that seems to correspond
        to this proper movement time.
        """
        processed_run_vec = np.full(run_vec.shape, np.nan)
        run = pd.Series(run_vec).diff().abs().rolling(int(self.framerate)).mean()
        processed_run_vec[run_vec > 0.035] = 1
        return processed_run_vec


@attr.s
class AnalogAnalysisTreadmillRows(AnalyzedAnalogTrace):
    """
    Analysis for the first version of the treadmill output,
    where each row was given a value, meaning the number of
    outputs is the number of frames times the number of rows.
    """

    num_of_lines = attr.ib(init=False)
    num_of_frames = attr.ib(init=False)

    def run(self):
        self.num_of_lines, self.num_of_frames = self._get_metadata()
        stim_and_juxta_vec = self._turn_analog_vec_into_per_frame(
            self.analog_trace.stimulus
        )
        stim_vec, juxta_vec = self._find_peaks(stim_and_juxta_vec)
        self.stim_vec = self._zero_to_nan(stim_vec)
        self.juxta_vec = self._zero_to_nan(juxta_vec)
        run_vec = self._turn_analog_vec_into_per_frame(self.analog_trace.run)
        run_vec = self.normalize_vec(run_vec)
        self.run_vec = self._populate_run(run_vec)
        self.spont_vec = self._populate_spont(stim_vec, juxta_vec)
        self.stand_vec = self._populate_stand()

        if self.occluder:
            self._populate_occluder()

        self._convert_to_series()

    def _turn_analog_vec_into_per_frame(self, vec: pd.Series) -> pd.Series:
        """
        Squeezes the input running data from a per-row basis to a per
        frame.
        """
        mean_data = vec.abs().rolling(self.num_of_lines).mean()
        data_per_frame = mean_data[self.num_of_lines - 1 :: self.num_of_lines]
        assert len(data_per_frame) == self.num_of_frames
        return data_per_frame

    def _populate_run(self, run_vec: np.ndarray) -> np.ndarray:
        """
        Processes the running analog data from the encoders.
        Looks for 1 second long periods in which the mouse ran
        by diffing the raw output of the encoder and checking if
        it passes some magic number that seems to correspond
        to this proper movement time.
        """
        processed_run_vec = np.full(run_vec.shape, np.nan)
        run = pd.Series(run_vec).diff().abs().rolling(int(self.framerate)).mean()
        processed_run_vec[run_vec > 0.035] = 1
        return processed_run_vec


@attr.s
class AnalogAnalysisMrduino(AnalyzedAnalogTrace):
    def run(self):
        stim_vec, juxta_vec = self._find_peaks()
        run_vec = self._populate_run()
        spont_vec = self._populate_spont(stim_vec, juxta_vec)
        if self.occluder:
            self._populate_occluder()

        self._init_vecs()
        self._fit_frames_to_analog(stim_vec, juxta_vec, run_vec, spont_vec)
        self._convert_to_series()


@attr.s
class AnalogAnalysisOld(AnalyzedAnalogTrace):
    def run(self):
        # Analog peak detection
        true_stim, juxta = self._find_peaks()
        stim_vec, juxta_vec = self._populate_stims(true_stim, juxta)
        run_vec = self._populate_run()
        spont_vec = self._populate_spont(stim_vec, juxta_vec)
        if self.occluder:
            self._populate_occluder()

        # Fit the analog vector to frame vector
        # self.__extract_time_series()
        self._init_vecs()
        self._fit_frames_to_analog(stim_vec, juxta_vec, run_vec, spont_vec)
        self._convert_to_series()

    def _find_peaks(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find the indices in which a peak occurred, and discern between peaks
        that were real stimuli and juxta peaks
        :return: Two numpy arrays with the stimulus and juxta peak indices
        """
        diffs_stim = np.where(self.analog_trace.stimulus > 4)[0]
        if len(diffs_stim) > 0:
            diffs_stim_con = np.concatenate((np.atleast_1d(diffs_stim[0]), diffs_stim))
            idx_true_stim = np.concatenate(
                (
                    np.atleast_1d(diffs_stim[0]),
                    diffs_stim[np.diff(diffs_stim_con) > self.sample_rate * 10],
                )
            )

        else:
            idx_true_stim = np.array([])
        diffs_juxta = np.where(self.analog_trace.stimulus > 2.2)[0]
        if len(diffs_juxta) > 0:
            diffs_juxta_con = np.concatenate(
                (np.atleast_1d(diffs_juxta[0]), diffs_juxta)
            )
            idx_juxta_full = np.concatenate(
                (
                    np.atleast_1d(diffs_juxta[0]),
                    diffs_juxta[np.diff(diffs_juxta_con) > self.sample_rate * 10],
                )
            )

            # Separate between stimulus and juxta pulses
            idx_juxta = []
            for val in idx_juxta_full:
                diff = np.abs(idx_true_stim - val)
                idx = np.where(diff < self.sample_rate)[0]
                if idx.size > 0:
                    continue
                else:
                    idx_juxta.append(val)
        else:
            idx_juxta = []
        return idx_true_stim, np.array(idx_juxta)

    def _populate_stims(
        self, true_stim: np.ndarray, juxta: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        For each peak, "mark" the following frames as the response frames for that
        stimulus
        :param true_stim: Indices for a stimulus
        :param juxta: Indices for a juxta stimulus
        :return: None
        """
        stim_vec = np.full(self.analog_trace.shape[0], np.nan)
        juxta_vec = np.full(self.analog_trace.shape[0], np.nan)
        for idx in true_stim:
            last_idx = int(
                idx + (self.response_window + self.buffer_after_stim) * self.sample_rate
            )
            stim_vec[idx:last_idx] = 1

        for idx in juxta:
            last_idx = int(
                idx + (self.response_window + self.buffer_after_stim) * self.sample_rate
            )
            juxta_vec[idx:last_idx] = 1

        return stim_vec, juxta_vec


if __name__ == "__main__":
    analog_file = pathlib.Path("/data/Amit_QNAP/WFA/Activity/WT_RGECO/B/20190804/WFA-FITC_RGECO_800nm_1040nm_256px_x25_mag3_stk3_20190804_00001_analog_min_max.txt")
    tif_file = pathlib.Path("/data/Amit_QNAP/WFA/Activity/WT_RGECO/B/20190804/WFA-FITC_RGECO_800nm_1040nm_256px_x25_mag3_stk3_20190804_00001.tif")
    analog = pd.read_csv(analog_file, header=None, names=["stimulus", "run"], index_col=False)
    fps = 58.24
    an_trace = analog_trace_runner(
        tif_filename=tif_file,
        analog_trace=analog,
        analog_type=AnalogAcquisitionType.TREADMILL,
        timestamps=np.arange(18000) / fps,
        framerate=fps,
        start_time="0",
        occluder=False,
        )
    an_trace.run()

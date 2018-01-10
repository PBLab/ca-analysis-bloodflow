import pandas as pd
import numpy as np
import attr
from attr.validators import instance_of
import re
from typing import Tuple
from tifffile import TiffFile


@attr.s(slots=True)
class AnalogTraceAnalyzer:
    """
    Fit and match the analog trace corresponding to the puffs and running of the mouse with the neuronal trace.

    Usage:
        AnalogTraceAnalyzer(tif_filename, analog_trace).run()
    """
    tif_filename = attr.ib(validator=instance_of(str))  # Timelapse (doesn't need to be separated)
    analog_trace = attr.ib(validator=instance_of(pd.DataFrame))  # .txt file from ScanImage
    response_window = attr.ib(default=0.5, validator=instance_of(float))  # sec
    stim_total_time = attr.ib(default=1., validator=instance_of(float))  # sec
    move_thresh = attr.ib(default=0.25, validator=instance_of(float))  # V
    sample_rate = attr.ib(default=1000, validator=instance_of(int))  # Hz
    stim_vec = attr.ib(init=False)
    juxta_vec = attr.ib(init=False)
    spont_vec = attr.ib(init=False)
    run_vec = attr.ib(init=False)
    timestamps = attr.ib(init=False)
    framerate = attr.ib(init=False)
    num_of_channels = attr.ib(init=False)

    def run(self):
        # Analog peak detection
        true_stim, juxta = self.__find_peaks()
        stim_vec, juxta_vec = self.__populate_stims(true_stim, juxta)
        run_vec = self.__populate_run()
        spont_vec = self.__populate_spont(stim_vec, juxta_vec)

        # Fit the analog vector to frame vector
        self.__extract_time_series()
        self.__init_vecs()
        self.__fit_frames_to_analog(stim_vec, juxta_vec, run_vec, spont_vec)
        self.__convert_to_series()

    def __find_peaks(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find the indices in which a peak occurred, and discern between peaks
        that were real stimuli and juxta peaks
        :return: Two numpy arrays with the stimulus and juxta peak indices
        """
        diffs_stim = np.where(self.analog_trace.stimulus > 4.9)[0]
        diffs_stim_con = np.concatenate((np.atleast_1d(diffs_stim[0]), diffs_stim))
        idx_true_stim = np.concatenate((np.atleast_1d(diffs_stim[0]),
                                        diffs_stim[np.diff(diffs_stim_con) > self.sample_rate * 10]))

        diffs_juxta = np.where(self.analog_trace.stimulus > 2.2)[0]
        diffs_juxta_con = np.concatenate((np.atleast_1d(diffs_juxta[0]), diffs_juxta))
        idx_juxta_full = np.concatenate((np.atleast_1d(diffs_juxta[0]),
                                         diffs_juxta[np.diff(diffs_juxta_con) > self.sample_rate * 10]))

        # Separate between stimulus and juxta pulses
        idx_juxta = []
        for val in idx_juxta_full:
            diff = np.abs(idx_true_stim - val)
            idx = np.where(diff < self.sample_rate)[0]
            if idx.size > 0:
                continue
            else:
                idx_juxta.append(val)
        return idx_true_stim, np.array(idx_juxta)

    def __populate_stims(self, true_stim: np.ndarray, juxta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        For each peak, "mark" the following frames as the response frames for that
        stimulus
        :param true_stim: Indices for a stimulus
        :param juxta: Indices for a juxta stimulus
        :return: None
        """
        stim_vec = np.zeros(self.analog_trace.shape[0], dtype=np.uint8)
        juxta_vec = np.zeros(self.analog_trace.shape[0], dtype=np.uint8)
        for idx in true_stim:
            last_idx = int(idx+(self.response_window + self.stim_total_time) * self.sample_rate)
            stim_vec[idx:last_idx] = 1

        for idx in juxta:
            last_idx = int(idx + (self.response_window + self.stim_total_time) * self.sample_rate)
            juxta_vec[idx:last_idx] = 1

        return stim_vec, juxta_vec

    def __populate_run(self) -> np.ndarray:
        """
        Wherever the analog voltage passes the threshold, assign a 1 value
        :return: None
        """
        run_vec = np.zeros(self.analog_trace.shape[0], dtype=np.uint8)
        run_vec[self.analog_trace.run > self.move_thresh] = 1
        return run_vec

    def __populate_spont(self, stim_vec: np.ndarray, juxta_vec: np.ndarray) -> np.ndarray:
        """
        Wherever the juxta and stim vectors are zero - write 1
        :return: None
        """
        all_stims = stim_vec + juxta_vec
        spont_vec = np.logical_not(all_stims).astype(np.uint8)
        return spont_vec

    def __convert_to_series(self):
        self.stim_vec = pd.Series(self.stim_vec)
        self.juxta_vec = pd.Series(self.juxta_vec)
        self.run_vec = pd.Series(self.run_vec)
        self.spont_vec = pd.Series(self.spont_vec)

    def __extract_time_series(self):
        with TiffFile(self.tif_filename) as f:
            d = f.scanimage_metadata
            ser = f.series[0]

        self.framerate = d['SI.hRoiManager.scanFrameRate']
        self.num_of_channels = len(d['SI.hChannels.channelsActive'])
        regex = re.compile(r'frameTimestamps_sec = ([\d.]+)')
        timestamps = []
        for page in ser.pages[::self.num_of_channels]:  # assuming that channel 1 contains the data
            desc = page.image_description.decode()
            timestamps.append(float(regex.findall(desc)[0]))
        self.timestamps = np.array(timestamps)

    def __init_vecs(self):
        self.stim_vec = np.zeros_like(self.timestamps, dtype=np.uint8)
        self.juxta_vec = np.zeros_like(self.timestamps, dtype=np.uint8)
        self.run_vec = np.zeros_like(self.timestamps, dtype=np.uint8)
        self.spont_vec = np.zeros_like(self.timestamps, dtype=np.uint8)

    def __fit_frames_to_analog(self, stim_vec: np.ndarray, juxta_vec: np.ndarray,
                               run_vec: np.ndarray, spont_vec: np.ndarray):
        samples_per_frame = int(np.ceil(self.sample_rate/self.framerate))
        starting_idx = np.linspace(0, len(stim_vec), num=len(self.timestamps), dtype=np.int64, endpoint=False)
        end_idx = starting_idx + samples_per_frame

        for frame_idx, (start, end) in enumerate(zip(starting_idx, end_idx)):
            self.stim_vec[frame_idx] = 1 if stim_vec[start:end].mean() > 0.5 else 0
            self.juxta_vec[frame_idx] = 1 if juxta_vec[start:end].mean() > 0.5 else 0
            self.run_vec[frame_idx] = 1 if run_vec[start:end].mean() > 0.5 else 0
            self.spont_vec[frame_idx] = 1 if spont_vec[start:end].mean() > 0.5 else 0


if __name__ == '__main__':
    npz_file = r'/data/David/602_new_baseline_imaging_201217/602_HYPO_DAY_0__EXP_STIM__FOV_2_00001_CHANNEL_1_results.npz'
    analog_file = r'/data/David/602_new_baseline_imaging_201217/602_HYPER_DAY_0__EXP_STIM__FOV_2(2)_mag_5_bidirectional_2048_512_30Hz_00001_analog.txt'
    data = np.load(npz_file)
    filename = r'/data/David/602_new_baseline_imaging_201217/602_HYPO_DAY_0__EXP_STIM__FOV_2_00001.tif'
    analog = pd.read_csv(analog_file, sep=r'\t', header=None,
                         names=['stimulus', 'run'])
    an_trace = AnalogTraceAnalyzer(tif_filename=filename, analog_trace=analog)
    an_trace.run()

import attr
from attr.validators import instance_of
import pathlib
import tifffile
import numpy as np
import re
from datetime import datetime
import os


@attr.s(slots=True)
class FluoMetadata:
    """ Simple datacontainer class to hold metadata of fluorescence recordings """

    fname = attr.ib(validator=instance_of(pathlib.Path))
    fps = attr.ib(default=58.24)  # framerate
    num_of_channels = attr.ib(default=1)
    start_time = attr.ib(default=0)
    id_reg = attr.ib(default=r'(^\d+?)_', validator=instance_of(str))
    day_reg = attr.ib(default=r'_DAY.+?(\d+)_', validator=instance_of(str))
    fov_reg = attr.ib(default=r'_FOV.+?(\d+)_', validator=instance_of(str))
    cond_reg = attr.ib(default=r'[0-9]_(HYP.+?)_DAY', validator=instance_of(str))
    timestamps = attr.ib(init=False)
    mouse_id = attr.ib(init=False)
    condition = attr.ib(init=False)
    day = attr.ib(init=False)
    fov = attr.ib(init=False)

    def get_metadata(self):
        self._get_si_meta()
        self.mouse_id = str(self._get_meta_using_regex(self.id_reg))
        self.day = int(self._get_meta_using_regex(self.day_reg))
        self.fov = int(self._get_meta_using_regex(self.fov_reg))
        self.condition = str(self._get_meta_using_regex(self.cond_reg)).upper()

    def _get_si_meta(self):
        """ Parse the metadata from the SI-generated file """
        try:
            with open(self.fname, 'rb') as f:
                si_meta, _, _ = tifffile.read_scanimage_metadata(f)
                self.fps = self._round_fps(float(si_meta['SI.hRoiManager.scanFrameRate']))
                save_chans = si_meta['SI.hChannels.channelSave']
                if type(save_chans) is int:
                    self.num_of_channels = 1
                else:
                    self.num_of_channels = len(save_chans)
                self.start_time = str(datetime.fromtimestamp(os.path.getmtime(str(self.fname))))
                try:
                    num_slices = si_meta['SI.actualNumSlice']
                except KeyError:
                    num_slices = si_meta['SI.numSlices']
                length = num_slices * si_meta['SI.framesPerSlice']
                self.timestamps = np.arange(length)/self.fps
        except (TypeError, KeyError, ValueError):
            fps = self.fps if self.fps else 58.2
            data = tifffile.imread(self.fname).shape[0]
            self.timestamps = np.arange(data) / fps

    def _get_meta_using_regex(self, reg: str):
        """ Parse the given regex from the filename """
        reg = re.compile(reg, re.IGNORECASE)
        try:
            result = reg.findall(str(self.fname.name))[0]
            if isinstance(result, tuple):
                if len(result[0]) == 0:  # first match was empty
                    return result[1]
                else:
                    return result[0]
            else:
                return result
        except IndexError:
            return -1

    def _round_fps(self, fps: float):
        """Due to minor fluctuations in the measured FPS in ScanImage,
        the resulting time coordinates can vary between otherwise
        identical datasets. These very minor changes cause issues when trying
        to concatenate two datasets that were sampled at almost exactly the
        same rate.
        The aim of this function is to round off the true FPS value to a
        unified one, shared between all recordings.
        """
        TRUE_FPS_VALUES = np.array([7.68, 15.24, 30.04, 58.24])
        idx_of_closest_value = np.abs(TRUE_FPS_VALUES - fps).argmin()
        return TRUE_FPS_VALUES[idx_of_closest_value]

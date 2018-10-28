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
    fps = attr.ib(default=15.24)  # framerate
    num_of_channels = attr.ib(default=1)
    start_time = attr.ib(default=0)
    id_reg = attr.ib(default=r'(^\d+?)_', validator=instance_of(str))
    day_reg = attr.ib(default=r'_DAY.+?(\d+)_', validator=instance_of(str))
    fov_reg = attr.ib(default=r'_FOV.+?(\d+)_', validator=instance_of(str))
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
        self.condition = 'Hyper' if 'HYPER' in str(self.fname) else 'Hypo'
    
    def _get_si_meta(self):
        """ Parse the metadata from the SI-generated file """
        try:
            with tifffile.TiffFile(str(self.fname)) as f:
                si_meta = f.scanimage_metadata
                self.fps = si_meta['FrameData']['SI.hRoiManager.scanFrameRate']
                self.num_of_channels = len(si_meta['FrameData']['SI.hChannels.channelsActive'])
                self.start_time = str(datetime.fromtimestamp(os.path.getmtime(str(self.fname))))
                length = len(f.pages)//self.num_of_channels
                self.timestamps = np.arange(length)/self.fps
        except TypeError:
            self.timestamps = None

    def _get_meta_using_regex(self, reg: str):
        """ Parse the given regex from the filename """
        reg = re.compile(reg)
        try:
            return reg.findall(str(self.fname.name))[0]
        except IndexError:
            return 999

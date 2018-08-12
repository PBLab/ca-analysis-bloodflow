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
    timestamps = attr.ib(init=False)
    mouse_id = attr.ib(init=False)
    condition = attr.ib(init=False)
    day = attr.ib(init=False)
    fov = attr.ib(init=False)

    def get_metadata(self):
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
        
        id_reg = re.compile(r'(^\d+?)_')
        self.mouse_id = id_reg.findall(str(self.fname.name))[0]
        self.condition = 'Hyper' if 'HYPER' in str(self.fname) else 'Hypo'

        day_reg = re.compile(r'_DAY.+?(\d+)_')
        try:
            self.day = int(day_reg.findall(str(self.fname.name))[0])
        except IndexError:
            self.day = 99

        fov_reg = re.compile(r'_FOV.+?(\d+)_')
        try:
            self.fov = int(fov_reg.findall(str(self.fname.name))[0])
        except IndexError:
            self.fov = 99

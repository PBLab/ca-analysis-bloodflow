import attr
from attr.validators import instance_of
import numpy as np
import enum
from scipy.stats import mode


class ConversionMethod(enum.Enum):
    """
    Types of conversion
    """
    RAW = 1
    DFF = 2
    RAW_SUBTRACT = 3
    NONE = 4


@attr.s(slots=True)
class RawTraceConverter:
    """
    Covnert a raw fluorescence trace into something more useful
    """
    conversion_method = attr.ib(validator=instance_of(ConversionMethod))
    raw_data = attr.ib(validator=instance_of(np.ndarray))
    data_before_offset = attr.ib(init=False)
    converted_data = attr.ib(init=False)
    num_of_rois = attr.ib(init=False)
    num_of_slices = attr.ib(init=False)

    def convert(self) -> np.ndarray:
        """
        Main conversion method
        :return np.ndarray: Dimensions neurons * time
        """
        self.__set_params()
        if self.conversion_method is ConversionMethod.RAW:
            self.__convert_raw()

        elif self.conversion_method is ConversionMethod.RAW_SUBTRACT:
            self.__convert_raw_subtract()

        elif self.conversion_method is ConversionMethod.DFF:
            self.__convert_dff()

        elif self.conversion_method is ConversionMethod.NONE:
            self.__convert_none()

        self.__add_offset()
        return self.converted_data

    def __set_params(self):
        self.num_of_rois = self.raw_data.shape[0]
        self.num_of_slices = self.raw_data.shape[1]

    def __convert_raw(self):
        """
        Change the raw trace to a normalized raw trace.
        :return: None
        """
        maxes = np.max(self.raw_data, axis=1).reshape((self.num_of_rois, 1))
        maxes = np.tile(maxes, self.num_of_slices)

        self.data_before_offset = self.raw_data / maxes

    def __convert_raw_subtract(self):
        """
        Subtract the minimal value from the stack and then normalize it.
        :return: None
        """
        mins = np.min(self.raw_data, axis=1).reshape((self.num_of_rois, 1))
        mins = np.tile(mins, self.num_of_slices)
        zeroed_data = self.raw_data - mins

        maxes = np.max(zeroed_data, axis=1).reshape((self.num_of_rois, 1))
        maxes = np.tile(maxes, self.num_of_slices)

        self.data_before_offset = zeroed_data / maxes

    def __convert_dff(self):
        """
        Subtract the minimal value and divide by the mode to receive a DF/F estimate.
        :return: None
        """
        mins = np.min(self.raw_data, axis=1).reshape((self.num_of_rois, 1))
        mins = np.tile(mins, self.num_of_slices)
        zeroed_trace = self.raw_data - mins + 1
        mods = mode(zeroed_trace, axis=1)[0].reshape((self.num_of_rois, 1))
        mods = np.tile(mods, self.num_of_slices)

        self.data_before_offset = (self.raw_data-mods) / mods

    def __convert_none(self):
        self.data_before_offset = self.raw_data / 4

    def __add_offset(self):
        """
        For easier visualization, add an offset to each trace
        :return:
        """
        offsets = np.arange(self.num_of_rois).reshape((self.num_of_rois, 1))
        offsets = np.tile(offsets, self.num_of_slices)

        self.converted_data = self.data_before_offset + offsets
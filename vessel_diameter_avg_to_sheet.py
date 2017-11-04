"""
__author__ = Hagai Hargil
"""
from pathlib import Path
from h5py import File
from os.path import splitext
import attr
from attr.validators import instance_of
import numpy as np
from collections import namedtuple
import pandas as pd




def main(foldername, glob_str):
    all_files = Path(foldername).rglob(glob_str)
    for file in all_files:
        rat_trace = Rat(file=str(file))



@attr.s
class Rat(object):
    """
    Container for vessel diameter data and before\after MCAO
    """
    file = attr.ib(validator=instance_of(str))
    struct_name = attr.ib(default='mv_mpP')
    name = attr.ib(init=False)
    num_of_vessels = attr.ib(init=False)
    is_before_mca = attr.ib(init=False)
    vessel_data = attr.ib(init=False)
    img = attr.ib(init=False)

    def run(self):
        """
        Run the full class pipeline
        """
        self.__load()
        self.__populate_dataframe()
        self.push_to_sheet()

    def __load(self):
        """
        Read the data in the .mat file into the class
        """
        Line = namedtuple('Line', ('x1', 'x2', 'y1', 'y2'))
        with File(self.file, driver='core') as f:
            self.data = f[self.struct_name]
            self.img = np.array(f[self.data['first_frame'][0][0]]).T
            self.num_of_vessels = self.data['Vessel'].shape[0]
            self.diameter_data = np.array(f[self.data['Vessel'][:, 0]]['diameter'])
            self.line_x = np.array(f[self.data['Vessel'][:, 0]])
            self.line_y = np.array(f[self.data['Vessel'][:, 0]])


    def __populate_dataframe(self):
        """
        Create a dataframe containing the diameter and line data
        :return:
        """
        self.vessel_data = pd.DataFrame([], columns=['raw_data', 'x', 'y', 'mean_diameter', 'std'])
        for vessel_idx in range(self.num_of_vessels):
            cur_raw = self.vessel_data[vessel_idx, :]
            cur_mean = np.mean(cur_raw)
            cur_std = np.std(cur_raw)
            cur_line_x = self.line_x[vessel_idx, 0]['vessel_line/position/xy'][0, :]
            cur_line_y = self.line_y[vessel_idx, 0]['vessel_line/position/xy'][1, :]
            self.vessel_data.append(pd.DataFrame([[cur_raw, cur_line_x, cur_line_y, cur_mean, cur_std]]),
                                    ignore_index=True)

    def push_to_sheet(self):
        self.vessel_data.to_excel('RatData.xlsx')

if __name__ == '__main__':
    foldername = r'X:\David'
    glob_str = 'rat_*.mat'

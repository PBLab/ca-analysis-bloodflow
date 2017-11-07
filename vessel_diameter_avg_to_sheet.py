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
import re



def main(foldernames, glob_str):
    xl_writer = pd.ExcelWriter('RatData.xlsx')
    for folder in foldernames:
        all_files = Path(folder).rglob(glob_str)
        for file in all_files:
            rat_trace = Rat(file=str(file),
                            excel_writer=xl_writer)
            rat_trace.run()
    xl_writer.save()


@attr.s
class Rat(object):
    """
    Container for vessel diameter data and before\after MCAO
    """
    file = attr.ib(validator=instance_of(str))
    struct_name = attr.ib(default='mv_mpP')
    excel_writer = attr.ib(default=pd.ExcelWriter('RatData.xlsx'))
    name = attr.ib(init=False)
    num_of_vessels = attr.ib(init=False)
    is_before_mca = attr.ib(init=False)
    vessel_data = attr.ib(init=False)

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
            self.name = str(Path(self.file))
            try:
                self.num_of_vessels = f[self.struct_name]['Vessel'].shape[0]
            except AttributeError:
                self.num_of_vessels = 1
            self.diameter_data = []
            self.line_x = []
            self.line_y = []
            for idx in range(self.num_of_vessels):
                try:
                    self.diameter_data.append(np.array(f[f[self.struct_name]['Vessel'][idx, 0]]['diameter']))
                    self.line_x.append(f[f[self.struct_name]['Vessel'][idx, 0]]['vessel_line/position/xy'][0, :])
                    self.line_y.append(f[f[self.struct_name]['Vessel'][idx, 0]]['vessel_line/position/xy'][1, :])
                except AttributeError:
                    self.diameter_data.append(np.array(f[self.struct_name]['Vessel']['diameter']))
                    self.line_x.append(np.array(f[self.struct_name]['Vessel']['vessel_line/position/xy'][0, :]))
                    self.line_y.append(np.array(f[self.struct_name]['Vessel']['vessel_line/position/xy'][1, :]))


    def __populate_dataframe(self):
        """
        Create a dataframe containing the diameter and line data
        :return:
        """
        columns = ['raw_data', 'x', 'y', 'mean_diameter', 'std']
        self.vessel_data = pd.DataFrame([], columns=columns)
        for vessel_idx in range(self.num_of_vessels):
            cur_raw = np.squeeze(self.diameter_data[vessel_idx])
            cur_mean = np.mean(cur_raw)
            cur_std = np.std(cur_raw)
            cur_line_x = self.line_x[vessel_idx]
            cur_line_y = self.line_y[vessel_idx]
            self.vessel_data = self.vessel_data.append(pd.DataFrame([[cur_raw, cur_line_x, cur_line_y, cur_mean, cur_std]],
                                                                    columns=columns),
                                                       ignore_index=True)

    def push_to_sheet(self):
        sheet_name = self.name.replace('\\', '.')[9:]
        rat_num = sheet_name[:9]
        fov_reg = re.compile(r'_\d+.(\d)')
        fov_num = fov_reg.findall(sheet_name)[0] + '_'

        after_reg = re.compile('AFTER')
        try:
            is_after = after_reg.findall(self.name)[0]
        except IndexError:
            is_after = 'BEFORE'

        oldana = re.compile('[Oo]ldana')
        oldana_ = ''
        try:
            oldana_ = oldana.findall(self.name)[0]
        except IndexError:
            pass

        self.vessel_data.to_excel(self.excel_writer, sheet_name=rat_num + fov_num + is_after + oldana_)

if __name__ == '__main__':
    foldernames = [r'X:\David\rat_#100_280917', r'X:\David\rat_#919_280917']
    glob_str = '*vessels_only_*.mat'
    main(foldernames, glob_str)

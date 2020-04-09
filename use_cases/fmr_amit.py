"""
This file shows how to generate dF/F plots of two FOVs. The plots show all
of the cell's dF/F, the running and evoked activity epochs, and simple
statistics calculated from the data.
"""
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from calcium_bflow_analysis.single_fov_analysis import SingleFovParser, SingleFovViz
from calcium_bflow_analysis.fluo_metadata import FluoMetadata
from calcium_bflow_analysis.analog_trace import AnalogAcquisitionType


baseline_folder = pathlib.Path('/data/Amit_QNAP/Calcium_FXS/x10')
basic_fmr_filename = 'FXS_614_X10_FOV5_mag3_20181010_00005'
fmr_tif = baseline_folder / 'FXS_614' / f'{basic_fmr_filename}.tif'
fmr_results = fmr_tif.with_name(f'{basic_fmr_filename}_results.npz')
fmr_analog = fmr_tif.with_name(f'{basic_fmr_filename}_analog.txt')

basic_wt_filename = 'WT_674_X10_FOV4_mag3_20181009_00004'
wt_tif = baseline_folder / 'WT_674' / f'{basic_wt_filename}.tif'
wt_results = wt_tif.with_name(f'{basic_wt_filename}_results.npz')
wt_analog = wt_tif.with_name(f'{basic_wt_filename}_analog.txt')

id_reg = r'^[A-Z]{2,3}_(\w+?)_X10'
day_reg = '(1)'
fov_reg = r'_FOV(\d)_'
cond_reg = '^([A-Z]){2, 3}_'

# fmr_meta = FluoMetadata(fmr_tif, 30.03, 1, 0, id_reg, day_reg, fov_reg, cond_reg)
# fmr_meta.get_metadata()
# fmr_fov = SingleFovParser(fmr_analog, fmr_results, fmr_meta, AnalogAcquisitionType.TREADROWS, True)
# fmr_fov.parse()

wt_meta = FluoMetadata(wt_tif, 30.03, 1, 0, id_reg, day_reg, fov_reg, cond_reg)
wt_meta.get_metadata()
wt_fov = SingleFovParser(wt_analog, wt_results, wt_meta, AnalogAcquisitionType.TREADROWS, True)
wt_fov.parse()

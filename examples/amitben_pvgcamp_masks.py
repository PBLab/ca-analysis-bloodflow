from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats

from calcium_bflow_analysis.calcium_over_time import FileFinder, CalciumAnalysisOverTime, FormatFinder
from calcium_bflow_analysis.analog_trace import AnalogAcquisitionType
from calcium_bflow_analysis.calcium_trace_analysis import (
    CalciumReview,
    AvailableFuncs,
    plot_single_cond_per_mouse,
    filter_da,
)
from calcium_bflow_analysis.dff_analysis_and_plotting.dff_analysis import (
    calc_total_auc_around_spikes,
    calc_mean_auc_around_spikes,
    calc_mean_spike_num,
    calc_median_auc_around_spikes,
)


home = Path("/data/Amit_QNAP/PV-GCaMP/")
folder = Path("289")
results_folder = home / folder
assert results_folder.exists()
globstr = "289*.tif"
folder_and_files = {home / folder: globstr}
analog_type = AnalogAcquisitionType.TREADMILL
analog_format = FormatFinder('analog', '*analog.txt')
hdf5_format = FormatFinder('hdf5', '*.hdf5')
npz_fomrat = FormatFinder('caiman', '*results.npz')
colabeled_format = FormatFinder('colabeled', '*_colabeled.npy')

filefinder = FileFinder(
    results_folder=results_folder,
    folder_globs=folder_and_files,
    analog=analog_type,
    with_colabeled=False,
    filtered=False,
)

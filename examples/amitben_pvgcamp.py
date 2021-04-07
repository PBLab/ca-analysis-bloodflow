from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns

from calcium_bflow_analysis.calcium_over_time import FileFinder, CalciumAnalysisOverTime
from calcium_bflow_analysis.analog_trace import AnalogAcquisitionType
from calcium_bflow_analysis.calcium_trace_analysis import CalciumReview, AvailableFuncs, plot_single_cond_per_mouse, filter_da
from calcium_bflow_analysis.dff_analysis_and_plotting.dff_analysis import calc_auc, calc_mean_spike_num, calc_mean_dff


home = Path("/data/Amit_QNAP/PV-GCaMP/")
folder = Path("289")
results_folder = home / folder
assert results_folder.exists()
globstr = "*.tif"
folder_and_files = {home / folder: globstr}
analog_type = AnalogAcquisitionType.TREADMILL
filefinder = FileFinder(
    results_folder=results_folder,
    folder_globs=folder_and_files,
    analog=analog_type,
    with_colabeled=True,
)
# files_table = filefinder.find_files()
regex = {
    "cond_reg": r"(0)",
    "id_reg": r"(289)",
    "fov_reg": r"_FOV(\d)_",
    "day_reg": r"(0)"
}
# res = CalciumAnalysisOverTime(
#     files_table=files_table,
#     serialize=True,
#     folder_globs=folder_and_files,
#     analog=analog_type,
#     regex=regex,
# )
# res.run_batch_of_timepoints(results_folder)
# res.generate_ds_per_day(results_folder, '*1_colabeled.nc', recursive=True)
# res.generate_ds_per_day(results_folder, '*1_non_colabeled.nc', recursive=True)
non_ca = CalciumReview(results_folder, "*non_colabeled*.nc")

analysis_methods = [
    AvailableFuncs.AUC,
    AvailableFuncs.MEAN,
    AvailableFuncs.SPIKERATE,
]
epoch = "all"

colabeled = xr.open_dataset('/data/Amit_QNAP/PV-GCaMP/289/colabeled_data_of_day_0.nc')
non_colabeled = xr.open_dataset('/data/Amit_QNAP/PV-GCaMP/289/non_colabeled_data_of_day_0.nc')
colabeled_sel = filter_da(colabeled, epoch)
non_colabeled_sel = filter_da(non_colabeled, epoch)

# AUC
auc_colabeled = calc_auc(colabeled_sel)
colabeled_labels = np.full(auc_colabeled.shape, 'colabeled')
auc_non_colabeled = calc_auc(non_colabeled_sel)
non_colabeled_labels = np.full(auc_non_colabeled.shape, 'non_colabeled')
plots  = pd.DataFrame({'auc': np.concatenate([auc_colabeled, auc_non_colabeled]), 'labels': np.concatenate([colabeled_labels, non_colabeled_labels])})

# SR
sr_colabeled = calc_mean_spike_num(colabeled_sel)
sr_non_colabeled = calc_mean_spike_num(non_colabeled_sel)
plots['spikerate'] = np.concatenate([sr_colabeled, sr_non_colabeled])
plots_for_display = plots.drop(index=plots.auc.nlargest(3).index)

# Plotting
ax_auc = sns.swarmplot(data=plots_for_display, x='labels', y='auc', size=2.5)
ax_auc= sns.barplot(data=plots_for_display, x='labels', y='auc', ax=ax_auc, color="0.8")
ax_auc.set_title('Area under the curve')
plt.figure()
ax_sr = sns.swarmplot(data=plots_for_display, x='labels', y='spikerate')
ax_sr.set_title('Spike rate')

ax_auc.figure.savefig('/data/Hagai/pv_auc.pdf', transparent=True, dpi=300)

plt.show(block=True)

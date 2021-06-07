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
file_formats = [
    FormatFinder('analog', '*analog.txt'),
    FormatFinder('hdf5', '*.hdf5'),
    FormatFinder('caiman', '*results.npz'),
    FormatFinder('colabeled', '*_colabeled.npy'),
]
filefinder = FileFinder(
    results_folder=results_folder,
    file_formats=file_formats,
    folder_globs=folder_and_files,
)
files_table = filefinder.find_files()
regex = {
    "cond_reg": r"(0)",
    "id_reg": r"(289)",
    "fov_reg": r"_FOV(\d)_",
    "day_reg": r"(0)",
}
res = CalciumAnalysisOverTime(
    files_table=files_table,
    serialize=True,
    folder_globs=folder_and_files,
    analog=analog_type,
    regex=regex,
)
# res.run_batch_of_timepoints(results_folder)
# Run once the colabeled data and once the non-colabeled, and then rename them
# res.generate_ds_per_day(results_folder, '*1_colabeled.nc', recursive=True)
# res.generate_ds_per_day(results_folder, '*1_non_colabeled.nc', recursive=True)

home = Path("/data/Amit_QNAP/PV-GCaMP/")
folder = Path("289")
review_folder = home / folder
# non_ca = CalciumReview(review_folder, "*non_colabeled*.nc")

# analysis_methods = [
#     AvailableFuncs.AUC,
#     AvailableFuncs.MEAN,
#     AvailableFuncs.SPIKERATE,
# ]
epoch = "stim"

colabeled = xr.open_dataset("/data/Amit_QNAP/PV-GCaMP/289/colabeled_data_of_day_0.nc")
non_colabeled = xr.open_dataset(
    "/data/Amit_QNAP/PV-GCaMP/289/non_colabeled_data_of_day_0.nc"
)
colabeled_sel = filter_da(colabeled, epoch)
non_colabeled_sel = filter_da(non_colabeled, epoch)

# AUC
auc_colabeled = calc_mean_auc_around_spikes(colabeled_sel)
auc_colabeled_total = calc_total_auc_around_spikes(colabeled_sel)
auc_colabaled_median = calc_median_auc_around_spikes(colabeled_sel)
colabeled_labels = np.full(auc_colabeled.shape, "colabeled")
auc_non_colabeled = calc_mean_auc_around_spikes(non_colabeled_sel)
auc_non_colabeled_total = calc_total_auc_around_spikes(non_colabeled_sel)
auc_non_colabaled_median = calc_median_auc_around_spikes(non_colabeled_sel)
non_colabeled_labels = np.full(auc_non_colabeled.shape, "non_colabeled")
plots = pd.DataFrame(
    {
        "mean_auc": np.concatenate([auc_colabeled, auc_non_colabeled]),
        "labels": np.concatenate([colabeled_labels, non_colabeled_labels]),
        "total_auc": np.concatenate([auc_colabeled_total, auc_non_colabeled_total]),
        "median_auc": np.concatenate([auc_colabaled_median, auc_non_colabaled_median]),
    }
)

# SR
sr_colabeled = calc_mean_spike_num(colabeled_sel)
sr_non_colabeled = calc_mean_spike_num(non_colabeled_sel)
plots["spikerate"] = np.concatenate([sr_colabeled, sr_non_colabeled])
plots_for_display = plots.drop(index=plots.mean_auc.nlargest(0).index)

# Plotting
ax_auc = sns.boxenplot(
    data=plots_for_display, x="labels", y="mean_auc", showfliers=False
)
ax_auc = sns.barplot(
    data=plots_for_display, x="labels", y="mean_auc", ax=ax_auc, color="0.8"
)
ax_auc.set_title("Mean area under the curve per spike")
plt.figure()
ax_auc_tot = sns.boxenplot(
    data=plots_for_display, x="labels", y="total_auc", showfliers=False
)
ax_auc_tot = sns.barplot(
    data=plots_for_display, x="labels", y="total_auc", ax=ax_auc_tot, color="0.8"
)
ax_auc_tot.set_title("Total area under the curve")
plt.figure()
ax_auc_median = sns.barplot(data=plots_for_display, x="labels", y="median_auc", color="0.8")
ax_auc_median.set_title("Median AUC per spike")
plt.figure()
ax_sr = sns.boxenplot(
    data=plots_for_display, x="labels", y="spikerate", showfliers=False
)
ax_sr = sns.barplot(
    data=plots_for_display,
    x="labels",
    y="spikerate",
    ax=ax_sr,
    facecolor=(0.8, 0.8, 0.8, 0.6),
    errcolor=(0.9, 0.9, 0.9, 1),
)
ax_sr.set_title("Spike rate")

# Stats

print(
    "Mean AUC: ",
    scipy.stats.ttest_ind(
        plots.query('labels=="colabeled"').loc[:, "mean_auc"],
        plots.query('labels=="non_colabeled"').loc[:, "mean_auc"],
    ),
)

print(
    "Total AUC: ",
    scipy.stats.ttest_ind(
        plots.query('labels == "colabeled"').loc[:, "total_auc"],
        plots.query('labels == "non_colabeled"').loc[:, "total_auc"],
    ),
)

print(
    "Mean spike rate: ",
    scipy.stats.ttest_ind(
        plots.query('labels == "colabeled"').loc[:, "spikerate"],
        plots.query('labels == "non_colabeled"').loc[:, "spikerate"],
    ),
)

print(
    "Median AUC: ",
    scipy.stats.ttest_ind(
        plots.query('labels == "colabeled"').loc[:, "median_auc"],
        plots.query('labels == "non_colabeled"').loc[:, "median_auc"],
        nan_policy='omit',
    ),
)

ax_auc.figure.savefig("/data/Hagai/pv_auc.pdf", transparent=True, dpi=300)

plt.show(block=True)

"""Analyze the calcium data from the ChABC experiment.
"""

from pathlib import Path

import seaborn as sns
import matplotlib.pyplot as plt

from calcium_bflow_analysis.calcium_over_time import AnalogAcquisitionType, CalciumAnalysisOverTime, FileFinder, FormatFinder
from calcium_bflow_analysis.calcium_trace_analysis import CalciumReview, AvailableFuncs


#%% Parameter setup
results_folder = Path("/data/Amit_QNAP/Thy1GCaMP_chABC/")
assert results_folder.exists()
globstr = "[0-9]*.tif"
folder_and_files = {results_folder: globstr}
file_formats = [
    FormatFinder('analog', '*analog.txt'),
    FormatFinder('hdf5', '*.hdf5'),
    FormatFinder('caiman', '*results.npz'),
]
analog_type = AnalogAcquisitionType.TREADMILL

metadata_in_filenames = {
    "cond_reg": r"(Control)|(ABC)",
    "id_reg": r"^(\d+)_",
    "fov_reg": r"FOV(\d)",
    "day_reg": r"post(\d+)|(0)"
}

#%% Generate objects
filefinder = FileFinder(
    results_folder=results_folder,
    file_formats=file_formats,
    folder_globs=folder_and_files,
)
files_table = filefinder.find_files()
print(files_table)
res = CalciumAnalysisOverTime(
    files_table=files_table,
    serialize=True,
    folder_globs=folder_and_files,
    analog=analog_type,
    regex=metadata_in_filenames,
)

#%% Run analysis that generates .nc files
res.run_batch_of_timepoints(results_folder)
# res.generate_ds_per_day(results_folder, '*.nc', recursive=True)

#%% Analyze the aggregate .nc files

ca = CalciumReview(results_folder, "data_of_day_*.nc")

analysis_methods = [
    AvailableFuncs.AUC,
    AvailableFuncs.MEAN,
    AvailableFuncs.MEDIAN,
    AvailableFuncs.SPIKERATE,
]
epoch = "all"

summary_df = ca.apply_analysis_funcs_two_conditions(analysis_methods, epoch)

for measure, data in summary_df.groupby('measure'):
    fig, ax = plt.subplots()
    sns.lineplot(data=data, x='day', y='data', hue='condition', hue_order=['CONTROL', 'ABC'], markers=True, ax=ax)
    ax.set_title(f'Measure: {measure}')
    figure_name = measure.lstrip('calc_') + '.pdf'
    fig.savefig(results_folder / figure_name)

plt.show(block=False)

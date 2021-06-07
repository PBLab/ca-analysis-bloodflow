from pathlib import Path

from calcium_bflow_analysis.calcium_over_time import FormatFinder, FileFinder, CalciumAnalysisOverTime
from calcium_bflow_analysis.analog_trace import AnalogAcquisitionType


home = Path("/data/David/")
folder = Path("thy1_g_test")
results_folder = home / folder
assert results_folder.exists()
globstr = "*.tif"
folder_and_files = {home / folder: globstr}
file_formats = [
    FormatFinder('analog', '*analog.txt'),
    FormatFinder('hdf5', '*.hdf5'),
    FormatFinder('caiman', '*results.npz'),
]
filefinder = FileFinder(
    results_folder=results_folder,
    file_formats=file_formats,
    folder_globs=folder_and_files,
)
files_table = filefinder.find_files()
regex = {
    "cond_reg": r"exptype_(\w+)_hemi",
    "id_reg": r"mouse_(\w+)_test",
    "fov_reg": r"fov_(\d+)_condition",
    "day_reg": r"day_(\d)_fov"
}
analog_type = AnalogAcquisitionType.TREADMILL
res = CalciumAnalysisOverTime(
    files_table=files_table,
    serialize=True,
    folder_globs=folder_and_files,
    analog=analog_type,
    regex=regex,
)
res.run_batch_of_timepoints(results_folder)
# res.generate_ds_per_day(results_folder, '*.nc', recursive=True)

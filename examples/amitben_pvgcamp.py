from pathlib import Path

from calcium_bflow_analysis.calcium_over_time import FileFinder
from calcium_bflow_analysis.analog_trace import AnalogAcquisitionType


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
files_table = filefinder.find_files()
regex = {
    "cond_reg": r"(0)",
    "id_reg": r"(289)",
    "fov_reg": r"_FOV(\d)_",
    "day_reg": r"(0)"
}
print(files_table)
# res = CalciumAnalysisOverTime(
#     files_table=files_table,
#     serialize=True,
#     folder_globs=folder_and_files,
#     analog=analog_type,
#     regex=regex,
# )
# res.run_batch_of_timepoints(results_folder)
# res.generate_ds_per_day(results_folder, '*.nc', recursive=True)

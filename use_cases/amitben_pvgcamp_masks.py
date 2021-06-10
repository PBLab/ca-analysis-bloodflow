from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
import tifffile
import skimage.measure as measure
import h5py

from calcium_bflow_analysis.calcium_over_time import FileFinder, CalciumAnalysisOverTime, FormatFinder
from calcium_bflow_analysis.analog_trace import AnalogAcquisitionType


MOUSE_ID = "289"

home = Path("/data/Amit_QNAP/PV-GCaMP/")
folder = Path(f"{MOUSE_ID}_new_analysis")
results_folder = home / folder
assert results_folder.exists()
globstr = f"{MOUSE_ID}*.tif"
folder_and_files = {home / folder: globstr}
analog_type = AnalogAcquisitionType.TREADMILL
file_formats = [
    FormatFinder('analog', '*analog.txt'),
    FormatFinder('hdf5', '*.hdf5'),
    FormatFinder('caiman', '*results.npz'),
    FormatFinder('masked', '*_masked.tif'),
]
filefinder = FileFinder(
    results_folder=results_folder,
    file_formats=file_formats,
    folder_globs=folder_and_files,
)
file_table = filefinder.find_files()
print(f"Found {len(file_table)} files!")

all_pnn, all_non = [], []
fractions = []
for num, siblings in file_table.iterrows():
    mask = tifffile.imread(str(siblings.masked))
    labeled_mask = measure.label(mask)
    regions = pd.DataFrame(measure.regionprops_table(labeled_mask, properties=('label', 'area'))).set_index('label')
    print(f"Number of regions: {len(regions)}")
    with h5py.File(siblings.hdf5, 'r') as f:
        img_components = np.asarray(f['estimates']['img_components'])
        accepted_list = np.asarray(f['estimates']['accepted_list'])
    if len(accepted_list) > 0:
        print(f"We have {len(accepted_list)} accepted components out of {len(img_components)}")
        img_components = img_components[accepted_list]
    else:
        accepted_list = range(len(img_components))
    img_components[img_components > 0] = 1
    labeled_components = img_components * labeled_mask
    non_pnn_indices, pnn_indices = [], []
    assert len(accepted_list) == len(labeled_components) == len(img_components)
    for component_idx, single_labeled_component, single_component in zip(accepted_list, labeled_components, img_components):
        uniques, counts = np.unique(single_labeled_component, return_counts=True)
        if len(uniques) == 1:
            non_pnn_indices.append(component_idx)
            continue
        fraction_covered_by_pnn = counts[1] / single_component.sum() 
        fractions.append(fraction_covered_by_pnn)
        if fraction_covered_by_pnn < 0.1:
            non_pnn_indices.append(component_idx)
        if fraction_covered_by_pnn > 0.6:
            pnn_indices.append(component_idx)
        continue
    if len(pnn_indices) > 0:
        colabeled_fname = str(siblings.tif)[:-4] + '_colabeled.npy'
        np.save(colabeled_fname, np.asarray(pnn_indices))

    all_pnn.extend(pnn_indices)
    all_non.extend(non_pnn_indices)

print(f"found {len(all_pnn)} pnn cells in {len(file_table)} ROIs")
plt.hist(fractions)
plt.show()


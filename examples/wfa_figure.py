"""
Code for Amit's Calcium figure in his WFA article
"""
import pathlib

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

from calcium_bflow_analysis.dff_analysis_and_plotting.plot_cells_and_traces import show_side_by_side
from calcium_bflow_analysis.single_fov_analysis import SingleFovParser, SingleFovViz
from calcium_bflow_analysis.fluo_metadata import FluoMetadata
from calcium_bflow_analysis.analog_trace import AnalogAcquisitionType

foldername = pathlib.Path("/data/Hagai/WFA_InVivo/new/")
channel_suffix = '_CHANNEL_1.tif'
results_suffix = '_CHANNEL_1_results.npz'
tifs = [
    '774_WFA-FITC_RCaMP7_x10_mag4_1040nm_256px_FOV1_z200_200802_00001',
    '774_WFA-FITC_RCaMP7_x10_mag4_1040nm_256px_FOV1_z270_200802_00001',
    '774_WFA-FITC_RCaMP7_x10_FOV2_mag4_1040nm_256px_z275_200818_00001',
    '774_WFA-FITC_RCaMP7_x10_mag4_1040nm_256px_FOV2_z330_500802_00001',
]

channel1 = [(foldername / tif).with_name(tif + channel_suffix) for tif in tifs]
results = [(foldername / tif).with_name(tif + results_suffix) for tif in tifs]

pnn_coords = [
    np.array([5]),
    np.array([13]),
    np.array([10]),
    np.array([3, 6]),
]
non_pnn_coords = [
    np.array([4]),
    np.array([7]),
    np.array([11]),
    np.array([8]),
]
cell_radius = 6
fps = 58.2

id_reg = '^(774)'
day_reg = '(1)'
fov_reg = r'_FOV(\d)_'
cond_reg = '(1040)'

if __name__ == "__main__":
    fig_pnn = show_side_by_side(channel1, results, pnn_coords, cell_radius)
    fig_pnn.suptitle('PNN cells')
    fig_non_pnn = show_side_by_side(channel1, results, non_pnn_coords, cell_radius)
    fig_non_pnn.suptitle('Non-PNN cells')
    for idx, tif in enumerate(tifs):
        meta = FluoMetadata((foldername / tif).with_suffix('.tif'), fps,
                            1, 0, id_reg, day_reg, fov_reg, cond_reg)
        meta.get_metadata()
        analog_fname = (foldername / tif).with_name(tif + '_analog.txt')
        if not analog_fname.exists():
            continue
        single_fov = SingleFovParser(
            analog_fname,
            results[0],
            meta, AnalogAcquisitionType.TREADMILL, False)
        single_fov.parse()
        puff_coords: np.ndarray = single_fov.fluo_analyzed['epoch_times'].sel(epoch='stim').values
        diff = np.diff(np.concatenate([puff_coords, [0]]))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        legit_diffs = np.concatenate([[1000], np.diff(starts)]) > (fps * 12)
        starts = starts[legit_diffs]
        ends = ends[legit_diffs]
        ax_pnn = fig_pnn.axes[idx * 2 + 1]
        ax_non_pnn = fig_non_pnn.axes[idx * 2 + 1]
        for start, end in zip(starts, ends):
            ax_pnn.add_artist(matplotlib.patches.Rectangle((start / fps, 0), width=(end - start) / fps, height=10, facecolor=(0.1, 0.1, 0.1), alpha=0.5, edgecolor="None"))
            ax_non_pnn.add_artist(matplotlib.patches.Rectangle((start / fps, 0), width=(end - start) / fps, height=10, facecolor=(0.1, 0.1, 0.1), alpha=0.5, edgecolor="None"))

    plt.show(block=False)
    fig_pnn.savefig(foldername / 'pnn_cells.pdf', transparent=True, dpi=300)
    fig_non_pnn.savefig(foldername / 'non_pnn_cells.pdf', transparent=True, dpi=300)


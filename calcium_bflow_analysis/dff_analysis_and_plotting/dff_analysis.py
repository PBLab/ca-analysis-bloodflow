import pathlib
from typing import Tuple, List
import sys

import numpy as np
import pandas as pd
import xarray as xr
import seaborn as sns
from ansimarkup import ansiprint as aprint
import peakutils
import matplotlib
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.patches
import sklearn.metrics
import skimage.draw
import tifffile
import scipy.ndimage

from calcium_bflow_analysis import caiman_funcs_for_comparison
from calcium_bflow_analysis.colabeled_cells.find_colabeled_cells import TiffChannels

# from calcium_bflow_analysis.single_fov_analysis import SingleFovParser


def calc_dff(file) -> np.ndarray:
    """ Read the dF/F data from a specific file. If the data doesn't exist,
    caclulate it using CaImAn's function.
    """
    data = np.load(file)
    print(f"Analyzing {file}...")
    try:
        dff = data["F_dff"]
    except KeyError:
        dff = caiman_funcs_for_comparison.detrend_df_f_auto(
            data["A"], data["b"], data["C"], data["f"], data["YrA"]
        )
    finally:
        aprint(
            f"The shape of the <b>dF/F matrix</b> for file <i>{file}</i> is <yellow>{dff.shape}</yellow>."
        )

    return dff


def calc_dff_batch(files):
    """ Read data from a sequence of files """
    all_data = []
    for file in files:
        all_data.append(calc_dff(file))
    return np.concatenate(all_data)


def locate_spikes_peakutils(data, fps=30.03, thresh=0.75,
                            min_dist=None, max_allowed_firing_rate=1):
    """
    Find spikes from a dF/F matrix using the peakutils package.
    The fps parameter is used to calculate the minimum allowed distance \
    between consecutive spikes, and to disqualify cells which had no
    evident dF/F peaks, which result in too many false-positives.

    :param float max_allowed_firing_rate: Maximal number of spikes per second
    that are considered viable.
    """
    assert len(data.shape) == 2 and data.shape[0] > 0
    if min_dist is None:
        min_dist = int(fps)
    else:
        min_dist = int(min_dist)
    all_spikes = np.zeros_like(data)
    nan_to_zero = np.nan_to_num(data)
    max_spike_num = int(data.shape[1] // fps) * max_allowed_firing_rate
    for row, cell in enumerate(nan_to_zero):
        peaks = peakutils.indexes(cell, thres=thresh, min_dist=min_dist)
        num_of_peaks = len(peaks)
        if (num_of_peaks > 0) and (num_of_peaks < max_spike_num):
            all_spikes[row, peaks] = 1
    return all_spikes


def calc_mean_spike_num(data, fps=30.03, thresh=0.75):
    """
    Find the spikes in the data (using "locate_spikes_peakutils") and count
    them, to create statistics on their average number.
    :param data: Raw data, cells x time
    :param fps: Framerate
    :param thresh: Peakutils threshold for spikes
    :return: Number of spikes for each neuron
    """
    all_spikes = locate_spikes_peakutils(data, fps, thresh)
    mean_of_spikes = np.nanmean(all_spikes, axis=1)
    return mean_of_spikes


def scatter_spikes(
    raw_data, spike_data=None, downsample_display=10, time_vec=None, ax=None
):
    """
    Shows a scatter plots of spike locations on each individual fluorescent trace.
    Parameters:
        raw_data (np.ndarray): The original fluorescent traces matrix, cell x time.
        spike_data (np.ndarray): The result of the `locate_spikes` function, a matrix
                                 with 1 wherever a spike was detected, and 0 otherwise.
        downsample_display (int): Too many cells create clutter and are hard to display.
                                  This is the downsampling factor.
        time_vec (np.ndarray): 1D array with the x-axis values (time). If None, will
                               use simple range(0, max) integer values.
        ax (plt.Axes): Axes to plot the graph on. If none, the function will generate one.
    """

    if time_vec is None:
        time_vec = np.arange(raw_data.shape[1])
    if not ax:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    downsampled_data = raw_data[::downsample_display]
    num_displayed_cells = downsampled_data.shape[0]
    y_step = 2
    y_heights = np.arange(0, num_displayed_cells*y_step, y_step)[:, np.newaxis]
    ax.plot(time_vec, (downsampled_data + y_heights).T, linewidth=0.5)
    if spike_data is not None:
        peakvals = raw_data * spike_data
        peakvals[peakvals == 0] = np.nan
        peakvals = peakvals[::downsample_display]
        ax.plot(
            time_vec,
            (peakvals + y_heights).T,
            "r.",
            linewidth=0.1,
        )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Cell ID")
    yticklabels = ax.get_yticklabels()
    new_ticks = [str(idx) for idx, _ in enumerate(yticklabels)]
    ax.set_yticklabels(new_ticks)
    # print([lab for lab in yticklabels])
    # ax.yaxis.set_major_formatter(FormatStrFormatter("%d"))
    return fig, num_displayed_cells


def plot_mean_vals(
    data, x_axis=None, window=30, title="Rolling Mean", ax=None
) -> matplotlib.axes.Axes:
    """
    Calculate a mean rolling window on the data, after averaging the 0th (cell) axis.
    This can be used to calculate the rolling mean firing rate, if `data` is a 0-1 binary
    matrix containing spike locations, or the rolling mean dF/F value if `data` contains
    the raw dF/F values for all cells.
    Parameters:
        data (np.ndarray): Data to be rolling-windowed.
        x_axis (np.ndarray): 1D array of time points for display purposes.
        window (int): size of rolling window in number of array cells.
        title (str): Title of the figure.
        ax (plt.Axis): Axis to plot on. If None - creates a new one.

    Returns:
        ax (plt.Axis): Axis that was plotted on.
        mean (float): The mean value of the entire rolling data.
    """
    if x_axis is None:
        x_axis = np.arange(data.shape[1])
    mean = pd.DataFrame(data.mean(axis=0))
    mean["x"] = x_axis
    mean_val = mean.rolling(window=window).mean()
    ax = mean_val.plot(x="x", ax=ax)
    ax.set_xlabel("Time")
    ax.set_ylabel("Mean rate")
    ax.set_title(title)
    return ax, mean_val[0].mean()


def calc_auc(data):
    """ Return the normalized area under the curve of all neurons in the data matrix.
    Uses a simple trapezoidal rule, and subtracts the offset of each cell before
    the computation.
    """
    offsets = np.nanmin(data, axis=1)[:, np.newaxis]
    no_offset = data - offsets
    auc = np.nanmean(no_offset, axis=1)
    return auc


def calc_mean_dff(data):
    """ Return the mean dF/F value, and the SEM of all neurons in the data matrix.
    Subtracts the offset of each cell before the computation.
    """
    min_vec = np.atleast_2d(data.min(axis=1)).T
    data_no_offset = data - min_vec
    return np.nanmean(data_no_offset, axis=1)


def deinterleave(fname: str, data_channel: int, num_of_channels: int = 2):
    """ Takes a multichannel TIF and writes back to disk the channel with
    the relevant data. """
    data_pre_split = tifffile.imread(fname)
    new_fname = fname[:-4] + f"_CHANNEL_{data_channel}.tif"
    try:
        tifffile.imsave(
            new_fname, data_pre_split[data_channel - 1 :: num_of_channels], bigtiff=True
        )
    except PermissionError:
        raise
    return new_fname


if __name__ == "__main__":
    foldername = pathlib.Path("/data/Amit_QNAP/WFA/Activity/WT_RGECO/B/")
    results_file = (
        foldername / "B_WFA-FITC_RGECO_1040nm_x25_mag4_256_20190722_00001_results.npz"
    )
    tif = (
        foldername / "B_WFA-FITC_RGECO_1040nm_x25_mag4_256_20190722_00001.tif"
    )
    cell_radius = 9
    number_of_channels = 2
    fps = 58.24
    raw_data = np.load(results_file, allow_pickle=True)['F_dff']
    spikes = locate_spikes_peakutils(raw_data, fps)
    time_vec = np.arange(raw_data.shape[1]) / fps
    scatter_spikes(raw_data, spikes, downsample_display=1, time_vec=time_vec)
    plt.show()




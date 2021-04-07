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
import scipy.signal

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


def locate_spikes_peakutils(
    data, fps=30.03, thresh=0.8, min_dist=None, max_allowed_firing_rate=1
) -> np.ndarray:
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
    all_spikes: np.ndarray = np.zeros_like(data)
    nan_to_zero = np.nan_to_num(data)
    max_spike_num = int(data.shape[1] // fps) * max_allowed_firing_rate
    for row, cell in enumerate(nan_to_zero):
        peaks = peakutils.indexes(cell, thres=thresh, min_dist=min_dist)
        num_of_peaks = len(peaks)
        if (num_of_peaks > 0) and (num_of_peaks < max_spike_num):
            all_spikes[row, peaks] = 1
    return all_spikes


def locate_spikes_scipy(
    data, fps=30.03, thresh=0.8, min_dist=None, max_allowed_firing_rate=1
):
    """Find spikes from a dF/F matrix using the find_peaks function.
    The fps parameter is used to calculate the minimum allowed distance
    between consecutive spikes, and to disqualify cells which had no
    evident dF/F peaks, which result in too many false-positives.

    Parameters
    ----------
    data : np.ndarray
        (cell x time) np.ndarray which contains the dF/F data
    max_allowed_firing_rate : float
        Maximal number of spikes per second that are considered viable.
    fps : float, optional
        Frame rate
    thresh : float, optional
        Spike-finding threshold between (0, 1)
    min_dist : int or None, optional
        Number of measurements between consecutive spikes
    max_allowed_firing_rate : float, optional
        Maximal number of spikes. Used to filter out cells with "too many
        spikes".
    """
    assert len(data.shape) == 2 and data.shape[0] > 0
    sec_in_samples = int(fps)
    if min_dist is None:
        min_dist = sec_in_samples
    else:
        min_dist = int(min_dist)
    all_spikes: np.ndarray = np.zeros_like(data)
    max_spike_num = int(data.shape[1] // fps) * max_allowed_firing_rate
    nan_to_zero = np.nan_to_num(data)
    for row, cell in enumerate(nan_to_zero):
        peaks, _ = scipy.signal.find_peaks(
            cell,
            prominence=thresh,
            distance=min_dist,
            wlen=sec_in_samples * 5,
            width=(sec_in_samples // 8, sec_in_samples * 10)
        )
        num_of_peaks = len(peaks)
        if (num_of_peaks > 0) and (num_of_peaks < max_spike_num):
            all_spikes[row, peaks] = 1
    return all_spikes


def calc_mean_spike_num(data, fps=30.03, thresh=0.8):
    """
    Find the spikes in the data (using "locate_spikes_peakutils") and count
    them, to create statistics on their average number.
    :param data: Raw data, cells x time
    :param fps: Framerate
    :param thresh: Peakutils threshold for spikes
    :return: Number of spikes for each neuron
    """
    all_spikes = locate_spikes_scipy(data, fps, thresh)
    mean_of_spikes = np.nansum(all_spikes, axis=1)
    print(f"Found a total of {mean_of_spikes.sum()} spikes, or {mean_of_spikes.mean()} per neuron")
    return mean_of_spikes


def calc_mean_spike_num_no_background(data, fps=30.03, thresh=0.8, q=20):
    """Find the spikes in the data and count them, but do that
    after removing some quantile that is treated as background.
    """
    data = _filter_backgroud_from_dff(data, q=q)
    return calc_mean_spike_num(data=data, fps=fps, thresh=thresh)


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
    downsampled_data = np.where(
        downsampled_data < np.float64(8), downsampled_data, np.float64(0)
    )
    num_displayed_cells = downsampled_data.shape[0]
    y_step = 2
    y_heights = np.arange(0, num_displayed_cells * y_step, y_step)[:, np.newaxis]
    ax.plot(time_vec, (downsampled_data + y_heights).T, linewidth=0.5)
    if spike_data is not None:
        peakvals = raw_data * spike_data
        peakvals[peakvals == 0] = np.nan
        peakvals = peakvals[::downsample_display]
        ax.plot(time_vec, (peakvals + y_heights).T, "r.", linewidth=0.1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Cell")
    new_ticks = []
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


def calc_auc(data, *args):
    """ Return the normalized area under the curve of all neurons in the data matrix.
    Uses a simple trapezoidal rule, and subtracts the offset of each cell before
    the computation.
    """
    offsets = np.nanmin(data, axis=1)[:, np.newaxis]
    no_offset = data - offsets
    auc = np.nanmean(no_offset, axis=1)
    return auc


def calc_total_auc_around_spikes(data, fps=58.21, thresh=0.8):
    """Calculates the area under the data curve for the data, but only around
    spike locations. This should give a better approximation of the data when
    there are very high or very low counts of spikes."""
    spikes = locate_spikes_scipy(data, fps, thresh)
    spikes_bloated = bloat_area_around_spikes(spikes, int(fps / 2))
    auc = np.zeros_like(data)
    auc[np.where(spikes_bloated)] = data[np.where(spikes_bloated)]
    return np.nansum(auc, axis=1)


def calc_mean_auc_around_spikes(data, fps=58.21, thresh=0.8):
    """Calculates the area under the data curve for the data, but only around
    spike locations. This should give a better approximation of the data when
    there are very high or very low counts of spikes."""
    spikes = locate_spikes_scipy(data, fps, thresh)
    spikes_per_neuron = spikes.sum(axis=1)
    spikes_bloated = bloat_area_around_spikes(spikes, int(fps // 2))
    auc = np.zeros_like(data)
    auc[np.where(spikes_bloated > 0)] = data[np.where(spikes_bloated > 0)]
    neurons_that_spiked = np.where(spikes_per_neuron > 0)[0]
    auc[neurons_that_spiked] /= spikes_per_neuron[neurons_that_spiked, None]
    return np.nanmean(auc, axis=1)


def bloat_area_around_spikes(spikes: np.ndarray, window: int) -> np.ndarray:
    """Increase the labeled area around each spike.
    Each spike is marked in the spikes array as 1, while the rest of the cells
    are 0. This function takes each spike and sets the area around it to have
    ones too."""
    win = np.ones(window)
    bloated = np.zeros_like(spikes)
    for idx, row in enumerate(spikes):
        bloated[idx] = np.clip(np.convolve(row, win, 'same'), 0, 1)

    return bloated


def calc_mean_dff(data, *args):
    """ Return the mean dF/F value, and the SEM of all neurons in the data matrix.
    Subtracts the offset of each cell before the computation.
    """
    min_vec = np.atleast_2d(np.nanmin(data, axis=1)).T
    data_no_offset = data - min_vec
    return np.nanmean(data_no_offset, axis=1)


def _filter_backgroud_from_dff(data: np.ndarray, q: int = 20) -> np.ndarray:
    """Filters out a quantile q from the data, returning it
    in the same shape but with values below q as nan.
    """
    q: np.ndarray = np.nanpercentile(data, 20, axis=1)
    above = data > q.reshape((len(q), 1))
    data[~above] = np.nan
    return data


def calc_mean_dff_no_background(data):
    """Calculates the mean dF/F value of the given data after getting rid
    of the lower quantiles of it.
    """
    filtered = _filter_backgroud_from_dff(data, q=20)
    return np.nanmean(filtered, axis=1)


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


def generate_spikes_roc_curve(dff: np.ndarray, fps: float):
    """
    To better assess the validity of a chosen threshold for spike detection
    in a dF/F trace, a ROC curve will be plotted. However, since we don't
    possess the spiking ground truth, we have to use a different heuristic
    in order to decide on the best threshold. For a given dF/F this function
    will try multiple spike thresholds and plot the resulting spiking rate,
    in hopes of identifying a region in which a change threshold doesn't
    significantly change the resulting spike rate.

    Parameters:
        :param np.ndarray dff: Cells x time
        :param float fps: Frames per second
    """
    threshold_boundaries = np.arange(0.4, 0.99, 0.05)
    spike_nums: np.ndarray = np.zeros_like(threshold_boundaries)
    for idx, thresh in enumerate(threshold_boundaries):
        spikes: np.ndarray = locate_spikes_peakutils(
            dff, fps, thresh=thresh, max_allowed_firing_rate=np.inf
        )
        spike_nums[idx] = spikes.sum()

    spike_nums /= dff.shape[0]
    spike_nums /= dff.shape[1] / fps
    fig, ax = plt.subplots()
    ax.plot(threshold_boundaries, spike_nums)
    ax.set_title("ROC for spike numbers as a function of the threshold")
    ax.set_xlabel("Threshold Value")
    ax.set_ylabel("# spikes / cell / second")


if __name__ == "__main__":
    baseline_folder = pathlib.Path("/data/Amit_QNAP/Calcium_FXS/x10")
    basic_fmr_filename = "FXS_614_X10_FOV5_mag3_20181010_00005"
    fmr_tif = baseline_folder / "FXS_614" / f"{basic_fmr_filename}.tif"
    fmr_results = fmr_tif.with_name(f"{basic_fmr_filename}_results.npz")
    fmr_analog = fmr_tif.with_name(f"{basic_fmr_filename}_analog.txt")

    # cell_radius = 9
    # number_of_channels = 2
    fps = 30.04
    raw_data = np.load(fmr_results, allow_pickle=True)["F_dff"]
    spikes = locate_spikes_scipy(raw_data, fps)
    time_vec = np.arange(raw_data.shape[1]) / fps
    scatter_spikes(raw_data, spikes, downsample_display=1, time_vec=time_vec)
    plt.show()


import numpy as np
import pandas as pd
import xarray as xr
from ansimarkup import ansiprint as aprint
import peakutils
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib import gridspec
import sklearn.metrics

import caiman_funcs_for_comparison


def calc_dff(file) -> np.ndarray:
    """ Read the dF/F data from a specific file. If the data doesn't exist,
    caclulate it using CaImAn's function.
    """
    data = np.load(file)
    print(f"Analyzing {file}...")
    try:
        dff =  data['F_dff']
    except KeyError:
        dff =  caiman_funcs_for_comparison.detrend_df_f_auto(data['A'], data['b'], data['C'],
                                                                data['f'], data['YrA'])
    finally:
        aprint(f"The shape of the <b>dF/F matrix</b> for file <i>{file}</i> is <yellow>{dff.shape}</yellow>.")

    return dff


def calc_dff_batch(files):
    """ Read data from a sequence of files """
    all_data = []
    for file in files:
        all_data.append(calc_dff(file))
    return np.concatenate(all_data)


def locate_spikes_peakutils(data, fps=30.03, thresh=0.65):
    """ 
    Find spikes from a dF/F matrix using the peakutils package.
    The fps parameter is used to calculate the minimum allowed distance \
    between consecutive spikes. 
    """
    assert len(data.shape) == 2 and data.shape[0] > 0
    all_spikes = np.zeros_like(data)
    min_dist = int(fps)
    for row, cell in enumerate(data):
        peaks = peakutils.indexes(cell, thres=thresh, min_dist=min_dist)
        all_spikes[row, peaks] = 1
    
    return all_spikes


def calc_mean_spike_rate(data, fps=30.03, thresh=0.65):
    """
    Find the spikes in the data (using "locate_spikes_peakutils") and count
    them, to create statistics on their average number.
    :param data: Raw data, cells x time
    :param fps: Framerate
    :param thresh: Peakutils threshold for spikes
    :return: mean, SEM of spike number for that given matrix.
    """
    all_spikes = locate_spikes_peakutils(data, fps, thresh)
    sum_of_spikes = all_spikes.sum(axis=1)
    mean_spike_rate = sum_of_spikes.mean()
    sem_spike_rate = sum_of_spikes.std(ddof=1) / np.sqrt(sum_of_spikes.shape[0])
    return mean_spike_rate, sem_spike_rate


def scatter_spikes(raw_data, spike_data, downsample_display=10, time_vec=None):
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
    """
    if time_vec is None:
        time_vec = np.arange(raw_data.shape[1])
    x, y = np.nonzero(spike_data)
    fig, ax = plt.subplots()
    downsample_display = 10
    num_displayed_cells = raw_data.shape[0] // downsample_display
    ax.plot(time_vec,
            (raw_data[:-10:downsample_display] + np.arange(num_displayed_cells)[:, np.newaxis]).T)
    peakvals = raw_data * spike_data
    peakvals[peakvals == 0] = np.nan
    ax.plot(time_vec,
            (peakvals[:-10:downsample_display] + np.arange(num_displayed_cells)[:, np.newaxis]).T,
            'r.', linewidth=0.1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Cell ID')
    return fig, num_displayed_cells


def plot_mean_vals(data, x_axis=None, window=30, title='Rolling Mean',
                   ax=None) -> matplotlib.axes.Axes:
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
    mean['x'] = x_axis
    mean_val = mean.rolling(window=window).mean()
    ax = mean_val.plot(x='x', ax=ax)
    ax.set_xlabel('Time')
    ax.set_ylabel('Mean rate')
    ax.set_title(title)
    return ax, mean_val[0].mean()


def calc_auc(data):
    """ Return the mean area under the curve, with its SEM of all neurons in the data matrix.
    Uses a simple trapezoidal rule, and subtracts the offset of each cell before 
    the computation.
    """
    x = np.arange(data.shape[1])
    all_auc = []
    for cell in data:
        no_offset = cell - cell.min()
        result = sklearn.metrics.auc(x, no_offset)
        all_auc.append(result)
    all_auc = np.array(all_auc)
    return all_auc.mean(), all_auc.std(ddof=1) / np.sqrt(all_auc.shape[0])


def calc_mean_dff(data):
    """ Return the mean dF/F value, and the SEM of all neurons in the data matrix.
    Subtracts the offset of each cell before the computation.
    """
    min_vec = np.atleast_2d(data.min(axis=1)).T
    data_no_offset = data - min_vec
    return data_no_offset.mean(), data_no_offset.std(ddof=1) / np.sqrt(data_no_offset.shape[0])
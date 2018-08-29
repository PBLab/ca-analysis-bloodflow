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


def calc_dff(file):
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
    return fig


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
    """
    if x_axis is None:
        x_axis = np.arange(data.shape[1])
    mean = pd.DataFrame(data.mean(axis=0))
    mean['x'] = x_ax
    ax = mean.rolling(window=window).mean().plot(x='x', ax=ax)
    ax.set_xlabel('Time')
    ax.set_ylabel('Mean rate')
    ax.set_title(title)
    return ax


def calc_auc(data):
    """ Return the total area under the curve of all neurons in the data matrix.
    Uses a simple trapezoidal rule, and subtracts the offset of each cell before 
    the computation.
    """
    summed_auc = 0
    x = np.arange(data.shape[1])
    for cell in data:
        no_offset = cell - cell.min()
        summed_auc += sklearn.metrics.auc(x, no_offset)
    return summed_auc


def calc_mean_dff(data):
    """ Return the mean dF/F value of all neurons in the data matrix.
    Subtracts the offset of each cell before the computation.
    """
    min_vec = np.atleast_2d(data.min(axis=1)).T
    data_no_offset = data - min_vec
    return data_no_offset.mean()
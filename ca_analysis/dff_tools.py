import pathlib
from typing import Tuple, List
import sys
sys.path.append('/data/MatlabCode/PBLabToolkit/CalciumDataAnalysis/python-ca-analysis-bloodflow')


import numpy as np
import pandas as pd
import xarray as xr
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

from ca_analysis import caiman_funcs_for_comparison
from ca_analysis.find_colabeled_cells import TiffChannels


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
        if len(peaks) > 0:
            all_spikes[row, peaks] = 1
        else:
            print("No spikes found.")
    
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
    mean_of_spikes = all_spikes.sum(axis=1) / data.shape[1]
    return mean_of_spikes


def scatter_spikes(raw_data, spike_data=None, downsample_display=10, time_vec=None):
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
    fig, ax = plt.subplots()
    downsampled_data = raw_data[::downsample_display]
    num_displayed_cells = downsampled_data.shape[0]
    ax.plot(time_vec,
            (downsampled_data + np.arange(num_displayed_cells)[:, np.newaxis]).T)
    if spike_data is not None:
        peakvals = raw_data * spike_data
        peakvals[peakvals == 0] = np.nan
        ax.plot(time_vec,
                (peakvals[::downsample_display] + np.arange(num_displayed_cells)[:, np.newaxis]).T,
                'r.', linewidth=0.1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Cell ID')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
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


def calc_auc(data, norm_factor=1):
    """ Return the normalized area under the curve of all neurons in the data matrix.
    Uses a simple trapezoidal rule, and subtracts the offset of each cell before 
    the computation.
    """
    x = np.arange(data.shape[1])
    all_auc = []
    for cell in data:
        no_offset = cell - cell.min()
        auc = sklearn.metrics.auc(x, no_offset)
        result = auc / (data.shape[1] * norm_factor)
        all_auc.append(result)
    all_auc = np.array(all_auc) 
    return all_auc

def calc_mean_dff(data):
    """ Return the mean dF/F value, and the SEM of all neurons in the data matrix.
    Subtracts the offset of each cell before the computation.
    """
    min_vec = np.atleast_2d(data.min(axis=1)).T
    data_no_offset = data - min_vec
    return data_no_offset.mean(1)


def display_heatmap(data, ax=None, epoch='All cells', downsample_factor=8,
                    fps=30.03):
    """ Show an "image" of the dF/F of all cells """
    if not ax:
        _, ax = plt.subplots()
    downsampled = data[::downsample_factor, ::downsample_factor].copy()
    top = np.nanpercentile(downsampled, q=95)
    bot = np.nanpercentile(downsampled, q=5)
    try:
        xaxis = np.arange(downsampled.shape[1]) * downsample_factor / fps
        yaxis = np.arange(downsampled.shape[0])
        ax.pcolor(xaxis, yaxis, downsampled, vmin=bot, vmax=top)
    except ValueError:  # emptry array
        return
    ax.set_aspect('auto')
    ax.set_ylabel('Cell ID')
    ax.set_xlabel('Time (sec)')
    ax.set_title(f"dF/F Heatmap for epoch {epoch}")


def extract_cells_from_tif(results_file: pathlib.Path, tif: pathlib.Path, 
                           indices=slice(None), num=20,
                           cell_radius=5, data_channel=TiffChannels.ONE,
                           number_of_channels=2,) -> Tuple[np.ndarray, float]:
    """ Load a raw TIF stack and extract an array of cells. The first dimension is
    the cell index, the second is time and the other two are the x-y images.
    Returns this 4D array, as well as the framerate of the acquisition.
    """
    res_data = np.load(results_file)
    if len(res_data['idx_components']) == len(res_data['crd']):  # new file
        coords = res_data['crd'][:num]
    else:
        relevant_indices = res_data['idx_components'][indices][:num]
        coords = res_data['crd'][relevant_indices]

    with tifffile.TiffFile(tif, movie=True) as f:
        data = f.asarray(slice(data_channel.value, None, number_of_channels))
        fps = f.scanimage_metadata['FrameData']['SI.hRoiManager.scanFrameRate']

    masks = extract_mask_from_coords(coords, data.shape[1:], cell_radius)
    cell_data = [data[:, mask[0], mask[1]] for mask in masks]
    return np.array(cell_data), fps


def extract_mask_from_coords(coords, img_shape, cell_radius) -> List[List[np.ndarray]]:
    """ Takes the coordinates ['crd' key] from a loaded results.npz file
    and extract masks around cells from it.
    Returns a list with a length of all detected cells. Each element in that
    list is a 2-element list containing two arrays with the row and column
    coordinates of that rectangle. To be used as data[mask[0], mask[1]].
    """
    coms_untouched = np.array([coords[idx]['CoM'] for idx in range(len(coords))], dtype=np.int16)
    cell_coms = np.clip(coms_untouched - cell_radius, 0, np.iinfo(np.int16).max)
    masks = [skimage.draw.rectangle(cell, extent=cell_radius*2, shape=img_shape) for cell in cell_coms]
    return masks


def display_cell_excerpts_over_time(results_file: pathlib.Path, tif: pathlib.Path, 
                                    indices=slice(None), num_to_display=20,
                                    cell_radius=5, data_channel=TiffChannels.ONE,
                                    number_of_channels=2, title='Cell Excerpts Over Time'):
    """ 
    Display cells as they fluoresce during the recording time, each cell in its
    own row, over time.
    Parameters:
    -----------
        results_file (pathlib.Path): Path to a results.npz file.
        tif (pathlib.Path): Path to the corresponding raw tiff recording.
        indices (slice or np.ndarray): List of indices of the relevant cells to look at.
        num_to_display (int): We usually have too many cells to display them all nicely.
        cell_radius (int): Number of pixels in the cell's radius.
        data_channel (Tiffchannels):  The channel containing the functional data.
        number_of_channels (int): Number of data channels.
    """
    cell_data, fps = extract_cells_from_tif(results_file, tif, indices, num_to_display,
                                            cell_radius, data_channel, number_of_channels)

    # Start plotting the cell excerpts, the first column is left currently blank
    idx_sample_start = np.linspace(start=0, stop=cell_data.shape[1], endpoint=False,
                                   num=num_to_display, dtype=np.uint64)
    idx_sample_end = idx_sample_start + np.uint64(20)
    w, h = matplotlib.figure.figaspect(1.)
    fig = plt.figure(figsize=(w, h))
    gs = gridspec.GridSpec(len(cell_data), num_to_display + 2, figure=fig, wspace=0.01, hspace=0.01)
    for row_idx, cell in enumerate(cell_data):
        ax_mean = plt.subplot(gs[row_idx, 0])
        mean_cell = cell.mean(0)
        vmin, vmax = mean_cell.min(), mean_cell.max()
        ax_mean.imshow(mean_cell, cmap='gray', vmin=vmin, vmax=vmax)
        ax_mean.set_xticks([])

        for col_idx, (frame_idx_start, frame_idx_end) in enumerate(zip(idx_sample_start,
                                                                       idx_sample_end), 2):
            ax = plt.subplot(gs[row_idx, col_idx])
            ax.imshow(cell[frame_idx_start:frame_idx_end, ...].mean(0), cmap='gray',
                               vmin=vmin, vmax=vmax)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)

    # Add labels to row and column at the edge
    for gs_idx, sample_idx in enumerate(idx_sample_start, 2):
        ax = plt.subplot(gs[-1, gs_idx])
        ax.set_xticks([cell_radius])
        label = f'{sample_idx/fps:.1f}'
        ax.set_xticklabels([label])
        ticklabel = ax.get_xticklabels()
        ticklabel[0].set_fontsize(6)
    
    for cell_idx in range(len(cell_data)):
        ax = plt.subplot(gs[cell_idx, 0])
        ax.set_yticks([cell_radius])
        ax.set_yticklabels([cell_idx+1])
    
    ax = plt.subplot(gs[-1, 0])
    ax.set_xlabel('Mean')
    ax.set_xticks([])
    
    fig.suptitle(title)
    fig.text(0.55, 0.04, 'Time (sec)', horizontalalignment='center')
    fig.text(0.04, 0.5, 'Cell ID', verticalalignment='center', rotation='vertical')
    fig.savefig(f'cell_mosaic_{title}.pdf', frameon=False, transparent=True)


def draw_rois_over_cells(fname: pathlib.Path, cell_radius=5):
    """ 
    Draw ROIs around cells in the FOV, and mark their number (ID).
    Parameters:
        fname (pathlib.Path): Deinterleaved TIF filename.
        cell_radius (int): Number of pixels in a cell's radius
    """
    assert fname.exists()
    try:
        results_file = next(fname.parent.glob(fname.name[:-4] + '*results.npz'))
    except StopIteration:
        print("Results file not found. Exiting.")
        return
    
    full_dict = np.load(results_file)
    if len(full_dict['idx_components']) == len(full_dict['crd']):
        rel_crds = full_dict['crd']
    else:
        rel_crds = full_dict['crd'][full_dict['idx_components']]
    fig, ax_img = plt.subplots()
    print("Reading TIF")
    data = tifffile.imread(str(fname)).mean(0)
    ax_img.imshow(data, cmap='gray')
    colors = [f'C{idx}' for idx in range(10)]
    masks = extract_mask_from_coords(rel_crds, data.shape, cell_radius)
    for idx, mask in enumerate(masks):
        origin = mask[1].min(), mask[0].min()
        rect = matplotlib.patches.Rectangle(origin, *mask[0].shape,
                                            edgecolor=colors[idx % 10], facecolor='none',
                                            linewidth=0.5)
        ax_img.add_patch(rect)
        ax_img.text(*origin, str(idx+1), color='w')


if __name__ == '__main__':
    # results_file = '/data/Amit_QNAP/WFA/Activity/WT_RGECO/522/940/522_WFA-FITC_RGECO_X25_mag3_stim_20181017_00003_CHANNEL_2_results.npz'
    tif = '/data/Amit_QNAP/WFA/Activity/WT_RGECO/522/940/522_WFA-FITC_RGECO_X25_mag3_stim_20181017_00003.tif'
    # tif = '/data/David/crystal_skull_TAC_180719/626_HYPER_DAY_0/626_HYPER_DAY_0__EXP_STIM__FOV_1_00001_CHANNEL_1.tif'
    data_channel = TiffChannels.TWO
    number_of_chans = 2
    # display_cell_excerpts_over_time(results_file=pathlib.Path(results_file),
    #                                 tif=pathlib.Path(tif),
    #                                 data_channel=data_channel,
    #                                 number_of_channels=number_of_chans)

    draw_rois_over_cells(pathlib.Path(tif), cell_radius=5)
    plt.show(block=True)

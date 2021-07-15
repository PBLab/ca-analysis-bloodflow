import pathlib
from typing import Tuple, List, Union, Dict, Optional
import sys

import numpy as np
import pandas as pd
import seaborn as sns
import imageio
from matplotlib.collections import LineCollection
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.patches
import tifffile
import skimage
import caiman as cm
import scipy.sparse
import h5py

from calcium_bflow_analysis.colabeled_cells.find_colabeled_cells import TiffChannels


def sort_spikes_by_stim(dff: np.ndarray, spikes: np.ndarray, stim: np.ndarray, fps: float) -> pd.DataFrame:
    """Finds the closest spike for each stim for each neuron."""
    stim_edges = np.concatenate((np.diff(np.nan_to_num(stim)), [0]))
    assert len(stim_edges) == dff.shape[1]
    ends_of_stim_idx = np.where(stim_edges == 1)[0]
    frame_diffs = np.full((dff.shape[0], len(ends_of_stim_idx)), np.nan)
    dff_diffs = frame_diffs.copy()
    spikes_idx = np.array(
        np.where(spikes == 1)
    )  # two rows, 'time' columns. First row is row indices, second row is column index (i.e. time).
    for idx, (stim_start, stim_end) in enumerate(
        zip(ends_of_stim_idx[:-1], ends_of_stim_idx[1:])
    ):
        cur_spikes = spikes_idx.astype(np.float64)
        cur_spikes[
            :, (cur_spikes[1, :] < stim_start) | (cur_spikes[1, :] > stim_end)
        ] = np.nan
        cur_spikes = (
            cur_spikes[np.isfinite(cur_spikes)].reshape((2, -1)).astype(np.int64)
        )
        cur_spikes_df = pd.DataFrame({"row": cur_spikes[0], "column": cur_spikes[1]})
        first_spikes = cur_spikes_df.groupby(by="row", as_index=True).min()
        frame_diffs[first_spikes.index, idx] = (first_spikes.column - stim_start) / fps
        dff_diffs[first_spikes.index, idx] = dff[
            first_spikes.index, first_spikes.column
        ]

    frame_diffs = (
        pd.DataFrame(frame_diffs).reset_index().rename(columns={"index": "Cell number"})
    )
    frame_diffs = pd.melt(
        frame_diffs,
        id_vars="Cell number",
        var_name="Stimulus number",
        value_name="Delay [sec]",
    )

    dff_diffs = (
        pd.DataFrame(dff_diffs).reset_index().rename(columns={"index": "Cell number"})
    )
    dff_diffs = pd.melt(
        dff_diffs,
        id_vars="Cell number",
        var_name="Stimulus number",
        value_name="dF/F at delay",
    )

    data = pd.concat(
        (frame_diffs, dff_diffs["dF/F at delay"].to_frame()), axis=1, sort=False
    )
    # Change a couple of things around for seaborn plottings
    data["Cell number"] = data["Cell number"].astype(np.int32).astype("category")
    data["Stimulus number"] = data["Stimulus number"].astype("category")
    return data


def rank_dff_by_stim(dff: np.ndarray, spikes: np.ndarray, stim: np.ndarray, fps: float):
    """Draws a plot of neurons ranked by the correlation they exhibit between
    a spike an air puff.

    Parameters:
    :param dff np.ndarray: Array of (cell x time) containing dF/F values of cells over time.
    :param spikes np.ndarray: Array of (cell x time) containing 1 wherever the cell fired
    and 0 otherwise. Result of ``locate_spikes_peakutils``.
    :param stim np.ndarray: A vector with the length of the experiment containing 1 wherever
    the stimulus occurred.
    :param float fps: Frames per second
    """
    data = sort_spikes_by_stim(dff, spikes, stim, fps)
    order = np.array(
        data.groupby("Cell number", sort=True).mean().sort_values("Delay [sec]").index
    )
    fig, ax_frame = plt.subplots()
    fig2, ax_dff = plt.subplots()
    fig3, ax_corr = plt.subplots()
    sns.barplot(
        data=data,
        x="Cell number",
        y="Delay [sec]",
        estimator=np.nanmean,
        ax=ax_frame,
        order=order,
    )
    sns.barplot(
        data=data,
        x="Cell number",
        y="dF/F at delay",
        estimator=np.nanmean,
        ax=ax_dff,
        order=order,
    )
    sns.scatterplot(data=data, x="Delay [sec]", y="dF/F at delay", ax=ax_corr)
    ax_frame.set_title(
        "Average minimal delay between spikes and stimulus for all neurons"
    )
    ax_dff.set_title("Average dF/F value of spike")


def show_side_by_side(
    tifs: List[pathlib.Path],
    results: List[pathlib.Path],
    crds: List[np.ndarray] = None,
    figsize=(36, 32),
    ax=None,
):
    """Shows a figure where each row is an image of a FOV,
    and all corresponding calcium traces. The image also draws
    a rectangle around each cell. The crds parameter allows you to
    choose which cells to display for each FOV.
    The ax parameter allows you to use a pre-existing axis, but it only works if
    you only have a single TIF to show and that axis has a shape of (1, 2).
    """
    assert len(tifs) == len(results)
    num_of_rows = len(tifs)

    if ax is None:
        fig, axes = plt.subplots(num_of_rows, 2, figsize=figsize)
        if num_of_rows == 1:
            axes = [axes]
    else:
        assert len(tifs) == 1
        axes = [ax]

    for tif, result, crd, ax in zip(tifs, results, crds, axes):
        with h5py.File(result, "r") as data:
            dff = np.asarray(data["estimates"]["F_dff"])
            if crd == 'NoneType':
                crd = np.arange(len(dff))
            dff = dff[crd]
            fps = data["params"]["data"]["fr"][()]
        dff = pd.DataFrame(dff.T).rolling(int(fps)).mean().to_numpy().T
        time_vec = np.arange(dff.shape[1]) / fps
        ax[0] = draw_rois_over_cells(tif, ax[0], crd, result)
        ax[1].plot(
            time_vec,
            (dff + np.arange(dff.shape[0])[:, np.newaxis]).T * 1,
            alpha=0.5,
            linewidth=2,
        )
        ax[1].spines["top"].set_visible(False)
        ax[1].spines["right"].set_visible(False)
        ax[1].set_xlabel("Time (seconds)")
        ax[1].set_ylabel("Cell ID")
        ax[1].yaxis.set_major_formatter(FormatStrFormatter("%d"))
        ax[1].set_yticks(np.arange(len(dff)))
        ax[1].set_yticklabels(crd)

    ax[0].figure.subplots_adjust(left=0.03, right=0.97, top=0.97, bottom=0.03)
    return ax[0].figure


def display_heatmap(data, ax=None, epoch="All cells", downsample_factor=8, fps=30.03, colorscale=None):
    """ Show an "image" of the dF/F of all cells """
    if not ax:
        _, ax = plt.subplots()
    if isinstance(downsample_factor, int):
        downsampled = data[::downsample_factor, ::downsample_factor].copy()
        xaxis = np.arange(downsampled.shape[1]) * downsample_factor / fps
    elif isinstance(downsample_factor, (tuple, list, np.ndarray)):
        assert len(downsample_factor) == 2
        downsampled = data[::downsample_factor[0], ::downsample_factor[1]].copy()
        xaxis = np.arange(downsampled.shape[1]) * downsample_factor[1] / fps

    if colorscale is None:
        top = np.nanpercentile(downsampled, q=95)
        bot = np.nanpercentile(downsampled, q=5)
    else:
        top, bot = colorscale
    try:
        yaxis = np.arange(downsampled.shape[0])
        ax.pcolor(xaxis, yaxis, downsampled, vmin=bot, vmax=top)
    except ValueError:  # emptry array
        return
    ax.set_aspect("auto")
    ax.set_ylabel("Cell ID")
    ax.set_xlabel("Time (sec)")
    ax.set_title(f"dF/F Heatmap for {epoch}")
    return (top, bot)


def extract_cells_from_tif(
    results_file: pathlib.Path,
    tif: pathlib.Path,
    indices=slice(None),
    num=20,
    cell_radius=5,
    data_channel=TiffChannels.ONE,
    number_of_channels=2,
) -> np.ndarray:
    """Load a raw TIF stack and extract an array of cells. The first dimension is
    the cell index, the second is time and the other two are the x-y images.
    Returns this 4D array.
    """
    res_data = np.load(results_file, allow_pickle=True)
    coords = res_data["crd"][indices][:num]

    with tifffile.TiffFile(tif, movie=True) as f:
        data = f.asarray()
        data = data[slice(data_channel.value, None, number_of_channels), ...]

    masks = extract_mask_from_coords(coords, data.shape[1:], cell_radius)
    cell_data = [data[:, mask[0], mask[1]] for mask in masks]
    return np.array(cell_data)


def extract_mask_from_coords(coords, img_shape, cell_radius) -> List[List[np.ndarray]]:
    """Takes the coordinates ['crd' key] from a loaded results.npz file
    and extract masks around cells from it.
    Returns a list with a length of all detected cells. Each element in that
    list is a 2-element list containing two arrays with the row and column
    coordinates of that rectangle. To be used as data[mask[0], mask[1]].
    """
    coms_untouched = np.array(
        [coords[idx]["CoM"] for idx in range(len(coords))], dtype=np.int16
    )
    cell_coms = np.clip(coms_untouched - cell_radius, 0, np.iinfo(np.int16).max)
    masks = [
        skimage.draw.rectangle(cell, extent=cell_radius * 2, shape=img_shape)
        for cell in cell_coms
    ]
    return masks


def display_cell_excerpts_over_time(
    results_file: pathlib.Path,
    tif: pathlib.Path,
    indices=slice(None),
    num_to_display=20,
    cell_radius=5,
    data_channel=TiffChannels.ONE,
    number_of_channels=2,
    fps=None,
    title="Cell Excerpts Over Time",
    output_folder=pathlib.Path("."),
):
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
        fps (float): Frames per second. Can be computed from the file.
    """
    cell_data = extract_cells_from_tif(
        results_file,
        tif,
        indices,
        num_to_display,
        cell_radius,
        data_channel,
        number_of_channels,
    )

    if not fps:
        with tifffile.TiffFile(str(tif), movie=True) as f:
            fps = f.scanimage_metadata["FrameData"]["SI.hRoiManager.scanFrameRate"]

    num_to_display = (
        len(cell_data) if len(cell_data) < num_to_display else num_to_display
    )

    # Start plotting the cell excerpts, the first column is left currently blank
    idx_sample_start = np.linspace(
        start=0,
        stop=cell_data.shape[1],
        endpoint=False,
        num=num_to_display,
        dtype=np.uint64,
    )
    idx_sample_end = idx_sample_start + np.uint64(20)
    fig = plt.figure(figsize=(18, 18))
    gs = gridspec.GridSpec(
        len(cell_data), num_to_display + 2, figure=fig, wspace=0.01, hspace=0.01
    )
    for row_idx, cell in enumerate(cell_data):
        ax_mean = plt.subplot(gs[row_idx, 0])
        mean_cell = np.nanmean(cell, axis=0)
        vmin, vmax = np.nanmin(mean_cell), np.nanmax(mean_cell)
        ax_mean.imshow(mean_cell.T, cmap="gray", vmin=vmin, vmax=vmax)
        ax_mean.set_xticks([])

        for col_idx, (frame_idx_start, frame_idx_end) in enumerate(
            zip(idx_sample_start, idx_sample_end), 2
        ):
            ax = plt.subplot(gs[row_idx, col_idx])
            ax.imshow(
                cell[frame_idx_start:frame_idx_end, ...].mean(0),
                cmap="gray",
                vmin=vmin,
                vmax=vmax,
            )
            ax.spines["top"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)

    # Add labels to row and column at the edge
    for gs_idx, sample_idx in enumerate(idx_sample_start, 2):
        ax = plt.subplot(gs[-1, gs_idx])
        ax.set_xticks([cell_radius])
        label = f"{sample_idx/fps:.1f}"
        ax.set_xticklabels([label])
        ticklabel = ax.get_xticklabels()
        ticklabel[0].set_fontsize(6)

    for cell_idx in range(len(cell_data)):
        ax = plt.subplot(gs[cell_idx, 0])
        ax.set_yticks([cell_radius])
        ax.set_yticklabels([cell_idx + 1])

    ax = plt.subplot(gs[-1, 0])
    ax.set_xlabel("Mean")
    ax.set_xticks([])

    fig.suptitle(title)
    fig.text(0.55, 0.04, "Time (sec)", horizontalalignment="center")
    fig.text(0.04, 0.5, "Cell ID", verticalalignment="center", rotation="vertical")
    fig.savefig(
        output_folder / f"cell_mosaic_{title}.pdf", frameon=False, transparent=True
    )


def get_coords_from_hdf5(hdf_fname: pathlib.Path) -> List[Dict]:
    """Extracts the coordinates of the detected components from the CaImAn
    results file."""
    with h5py.File(hdf_fname, "r") as f:
        a = f["estimates"]["A"]
        A = scipy.sparse.csc_matrix(
            (a["data"], a["indices"], a["indptr"]), shape=a["shape"]
        )
        dims = f["dims"][()]
    coordinates = cm.utils.visualization.get_contours(A, dims=dims)
    return coordinates


def _add_text_labels(ax: plt.Axes, coms: List[Tuple], roi_ids: np.ndarray):
    for com, roi_id in zip(coms, roi_ids):
        ax.text(com[0], com[1], str(roi_id), color="w", size=10, clip_on=True)


def get_accepted_components_idx(hdf_fname: pathlib.Path) -> Optional[np.ndarray]:
    """Returns the 0-based indices of the accepted components"""
    with h5py.File(hdf_fname, "r") as f:
        idx = f["estimates"]["accepted_list"][()]
    if idx == "NoneType":
        return None
    return idx


def _post_process_rois(all_rois_objects: List, crds: np.ndarray) -> Tuple[List[np.ndarray], List[Tuple]]:
    all_rois = []
    coms = []
    for crd in crds:
        current_roi = all_rois_objects[crd]["coordinates"]
        # current_roi[:, 0], current_roi[:, 1] = current_roi[:, 1], current_roi[:, 0].copy()
        all_rois.append(current_roi)
        comx = np.nanmean(current_roi[:, 0])
        comy = np.nanmean(current_roi[:, 1])
        coms.append((comx, comy))
    return all_rois, coms


def draw_rois_over_cells(
    tif_fname: Union[pathlib.Path, np.ndarray],
    ax_img=None,
    crds=None,
    results_file=None,
    roi_fname=None,
):
    """
    Draw ROIs around cells in the FOV, and mark their number (ID).
    Parameters:
        tif_fname (array or pathlib.Path): The image to show, or a deinterleaved TIF filename.
        ax_img (Axes): matplotlib Axes object to draw on. If None - will be created
        crds (List of ints): Specific indices of the cells to be shown. If None shows all.
        results_file(pathlib.Path): Path to the results file associated with the tif.
        roi_fname (pathlib.Path): Path to save a black image only with the ROIs
    """
    if isinstance(tif_fname, pathlib.Path):
        assert tif_fname.exists()
        if not results_file:
            try:
                results_file = next(
                    tif_fname.parent.glob(tif_fname.name[:-4] + "*results.npz")
                )
            except StopIteration:
                print("Results file not found. Exiting.")
            return
        tif = tifffile.imread(str(tif_fname)).mean(0)
    elif isinstance(tif_fname, np.ndarray):
        tif = tif_fname

    all_rois_objects = get_coords_from_hdf5(results_file)
    all_rois, coms = _post_process_rois(all_rois_objects, crds)

    if ax_img is None:
        fig, ax_img = plt.subplots()
    if roi_fname:
        new_size = tif.shape[0] / 100
        fig.set_size_inches((new_size, new_size))
    ax_img.imshow(np.zeros_like(tif), cmap="gray")
    ax_img.axis("off")
    ax_img.set_aspect("equal")
    ax_img.add_collection(LineCollection(all_rois, colors="white", linewidths=0.7))
    _add_text_labels(ax_img, coms, crds)

    if roi_fname:
        # ax_img.figure.savefig(str(roi_fname), transparent=True, format='tif', bbox_inches='tight', pad_inches=0)
        ax_img.figure.tight_layout(pad=0)
        ax_img.figure.canvas.draw()
        data = np.frombuffer(
            ax_img.figure.canvas.tostring_rgb(), dtype=np.uint8
        ).reshape((ax_img.figure.canvas.get_width_height()[::-1] + (3,)))
        imageio.imwrite(roi_fname, data)
        i = tifffile.imread(str(roi_fname))
        b = skimage.util.img_as_int(
            skimage.transform.resize(
                skimage.color.rgb2gray(i), tif.shape, anti_aliasing=True
            )
        )
        tifffile.imsave(str(roi_fname), b)
    else:
        ax_img.images.pop()
        ax_img.imshow(np.fliplr(np.flipud(tif)), cmap="gray")
    return ax_img


if __name__ == "__main__":
    tifs = [
        pathlib.Path(
            "/data/Amit_QNAP/WT_WFA-FITC_RCaMP_chABC/285/285_WT_RCaMP7_WFA-FITC_x10_mag3_FOV1_z490_1040nm_256px_210218_00001_CHANNEL_2.tif"
        ),
    ]
    results = [
        pathlib.Path(
            "/data/Amit_QNAP/WT_WFA-FITC_RCaMP_chABC/285/285_WT_RCaMP7_WFA-FITC_x10_mag3_FOV1_z200_1040nm_256px_210218_00001_CHANNEL_2_memmap__d1_256_d2_256_d3_1_order_C_frames_9000_.hdf5"
        ),
    ]
    fig = show_side_by_side(tifs, results)
    # fig.savefig('/data/Amit_QNAP/rcamp107_wfa_120320/all_fovs.pdf', transparent=True, dpi=300)
    plt.show(block=True)

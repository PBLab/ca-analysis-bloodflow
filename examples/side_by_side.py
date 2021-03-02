import warnings
import pathlib
import json
from typing import Optional

from magicgui import magicgui
import colorcet as cc
import skimage.transform
import skimage.exposure
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from caiman.utils.visualization import get_contours
import h5py

from calcium_bflow_analysis.dff_analysis_and_plotting import plot_cells_and_traces


def write_to_cache(foldername, data: dict):
    if not foldername.exists():
        return
    filename = foldername / "overlay_channels.json"
    try:
        with open(filename, "w") as f:
            json.dump(data, f)
    except (FileNotFoundError, PermissionError) as e:
        print(repr(e))


def read_from_cache(foldername) -> dict:
    if not foldername.exists():
        return
    filename = foldername / "overlay_channels.json"
    data = None
    try:
        with open(filename) as f:
            data = json.load(f)
    except (FileNotFoundError, PermissionError):
        pass
    return data


def _verify_fnames(*args):
    """Verify each given filename for existence. Raises ValueError if it
    doesn't."""
    for fname in args:
        if fname:
            if not fname.exists():
                raise ValueError


def _normalize_arrays(ch1: np.ndarray, ch2: np.ndarray):
    if ch1.shape == ch2.shape:
        return ch1, ch2
    if ch1.shape[0] < ch2.shape[0]:
        ch2 = skimage.transform.resize(
            ch2, ch1.shape, anti_aliasing=True, preserve_range=True
        )
    else:
        ch1 = skimage.transform.resize(
            ch1, ch2.shape, anti_aliasing=True, preserve_range=True
        )
    ch1 = skimage.exposure.rescale_intensity(ch1, out_range="int16").astype("int16")
    ch2 = skimage.exposure.rescale_intensity(ch2, out_range="int16").astype("int16")
    return ch1, ch2


def _process_single_channel_data(fname: pathlib.Path, frames: slice) -> np.ndarray:
    """Quick wrapper around basic IO"""
    data = tifffile.imread(str(fname))[frames]
    if data.ndim == 3:
        data = data.mean(axis=0)
    return data


def combine_two_images(ch1: np.ndarray, ch2: np.ndarray):
    """Used in case the GUI has to show an overlay of two channels"""
    ch1, ch2 = _normalize_arrays(ch1, ch2)
    im = ch1 * 0.5 + ch2 * 0.5
    return im, ch1, ch2


def two_channel_pipeline(
    ch1: np.ndarray,
    ch1_fname: pathlib.Path,
    ch2: np.ndarray,
    ch2_fname: pathlib.Path,
    fig,
):
    new_fname = ch1_fname.parent / (
        "combined_" + ch1_fname.stem + "_" + ch2_fname.stem + ".tif"
    )
    roi_fname = str(
        new_fname.parent
        / ("only_roi_" + ch1_fname.stem + "_" + ch2_fname.stem + ".tif")
    )
    new_fname = str(new_fname)
    tifffile.imwrite(new_fname, np.stack([ch1, ch2]))
    # ch1 -= ch1.min()
    # ch2 -= ch2.min()
    vmin1, vmax1 = ch1.min() * 1.1, ch1.max() * 0.9
    vmin2, vmax2 = ch2.min() * 1.1, ch2.max() * 0.9
    fig.axes[0].images.pop()
    fig.axes[0].imshow(ch1, cmap=cc.cm.kgy, vmin=vmin1, vmax=vmax1)
    fig.axes[0].imshow(ch2, cmap=cc.cm.kr, alpha=0.55, vmin=vmin2, vmax=vmax2)
    fig.axes[0].set_title("Ch1 is green, Ch2 is red")
    fig.canvas.set_window_title(f"{new_fname}")
    plt.show(block=False)
    return roi_fname


def _get_accepted_components(fname: pathlib.Path) -> np.ndarray:
    with h5py.File(fname, 'r') as f:
        data = f['estimates']['idx_components'][()]
        if len(data) == 0:
            data = np.arange(len(f['estimates']['F_dff'][()]))
    return data


def _determine_ch2_validity(ch2_fname: pathlib.Path) -> Optional[pathlib.Path]:
    if (ch2_fname.is_dir()) or (not ch2_fname.exists()) or (ch2_fname.suffix != ".tif"):
        warnings.warn(
            "The given filename for the other data channel was not found. Continuing without it."
        )
        return None
    return ch2_fname


@magicgui(
    call_button="Show",
    persist=True,
    result_widget=True,
    main_window=True,
    ch1_fname={"label": "Data file", "filter": "*.tif"},
    ch1_frames={"label": "Relevant data frames", "stop": 100_000},
    results_fname={"label": "CaImAn's HDF5", "filter": "*.hdf5"},
    ch2_fname={"label": "Overlay data with", "filter": "*.tif"},
    ch2_frames={"label": "Second channel frames", "stop": 100_000},
)
def show_traces_and_rois(
    ch1_fname: pathlib.Path,
    ch1_frames: slice,
    results_fname: pathlib.Path,
    ch2_fname: pathlib.Path,
    ch2_frames: slice,
):
    """Shows calicum traces and cell ROIs with a possible overlay of a second
    color.

    This tool is meant to assist in viewing the results of the CaImAn pipeline
    by showing a max projection of the data with the detected ROIs overlayed,
    and with the calcium traces presented at its side. Additionally the user
    may also provide a second color channel that will be overlayed on top of
    the first if that is of help.

    In addition to that, only specific frames from the recording can be
    included if the data contains two channels which aren't interleaved, or in
    other extraordinary cases.

    Parameters
    ----------
    ch1_fname : pathlib.Path
        Main data channel to use as the baseline image, usually the calcium
        activity data
    ch1_frames : slice
        Frames to take from the main data channel
    results_fname : pathlib.Path
        The CaImAn-generated HDF5 results file
    ch2_fname : pathlib.Path, optional
        A second color to overlay on top of the first, optional
    ch2_frames : slice, optional
        Frames to take from the second color channel
    """
    try:
        _verify_fnames(ch1_fname, ch2_fname, results_fname)
    except ValueError:
        return "Filepath error, please re-try"
    ch1 = _process_single_channel_data(ch1_fname, ch1_frames)
    ch2_fname = _determine_ch2_validity(ch2_fname)
    if ch2_fname:
        ch2 = _process_single_channel_data(ch2_fname, ch2_frames)
        im, ch1, ch2 = combine_two_images(ch1, ch2)
    else:
        im = ch1
    accepted_component_indices = _get_accepted_components(results_fname)
    fig = plot_cells_and_traces.show_side_by_side([im], [results_fname], [accepted_component_indices])
    if ch2_fname:
        roi_fname = two_channel_pipeline(ch1, ch1_fname, ch2, ch2_fname, fig)
    else:
        roi_fname = ch1_fname.parent / ("only_roi" + ch1_fname.name)

    plot_cells_and_traces.draw_rois_over_cells(
        im, results_file=results_fname, roi_fname=roi_fname, crds=accepted_component_indices,
    )
    plt.show(block=False)
    return roi_fname


if __name__ == "__main__":
    show_traces_and_rois.show(run=True)
    # show_traces_and_rois()

from calcium_bflow_analysis.dff_analysis_and_plotting.plot_cells_and_traces import draw_rois_over_cells
import pathlib
import json

from magicgui import magicgui, event_loop
import colorcet as cc
import skimage.transform
import skimage.exposure
import cv2
import tifffile
import numpy as np
import matplotlib.pyplot as plt

from calcium_bflow_analysis.dff_analysis_and_plotting import plot_cells_and_traces

# linux only
CACHE_FOLDER = pathlib.Path.home() / pathlib.Path('.cache/ca_analysis_bloodflow')
CACHE_FOLDER.mkdir(mode=0o777, parents=True, exist_ok=True)


def write_to_cache(foldername, data: dict):
    if not foldername.exists():
        return
    filename = foldername / 'overlay_channels.json'
    try:
        with open(filename, 'w') as f:
            json.dump(data, f)
    except (FileNotFoundError, PermissionError) as e:
        print(repr(e))


def read_from_cache(foldername) -> dict:
    if not foldername.exists():
        return
    filename = foldername / 'overlay_channels.json'
    data = None
    try:
        with open(filename) as f:
            data = json.load(f)
    except (FileNotFoundError, PermissionError):
        pass
    return data


def _find_start_end_frames(inp: str):
    splitted = inp.split(',')
    if len(splitted) == 1:
        return slice(None, None)
    try:
        start = int(splitted[0])
    except ValueError:
        start = None
    try:
        end = int(splitted[1])
    except ValueError:
        end = None
    return slice(start, end)


def _normalize_arrays(ch1: np.ndarray, ch2: np.ndarray):
    if ch1.shape == ch2.shape:
        return ch1, ch2
    if ch1.shape[0] < ch2.shape[0]:
        ch2 = skimage.transform.resize(ch2, ch1.shape, anti_aliasing=True, preserve_range=True)
    else:
        ch1 = skimage.transform.resize(ch1, ch2.shape, anti_aliasing=True, preserve_range=True)
    ch1 = skimage.exposure.rescale_intensity(ch1, out_range='int16').astype('int16')
    ch2 = skimage.exposure.rescale_intensity(ch2, out_range='int16').astype('int16')
    return ch1, ch2


@magicgui(call_button="Show", layout="form", ch1_fname={'fixedWidth': 1000})
def overlay_channels_and_show_traces(ch1_fname: str = ".tif", ch1_frames: str = "", ch2_fname: str = ".tif", ch2_frames: str = "", results_fname: str = ".npz", cell_radius: int = 6):
    ch1_fname = pathlib.Path(ch1_fname)
    if not ch1_fname.exists():
        return "Channel 1 path doesn't exist"
    ch2_fname = pathlib.Path(ch2_fname)
    if not ch2_fname.exists():
        return "Channel 2 path doesn't exist"
    results_fname = pathlib.Path(results_fname)
    if not results_fname.exists():
        return "Results path doesn't exist"
    ch1_slice = _find_start_end_frames(ch1_frames)
    ch2_slice = _find_start_end_frames(ch2_frames)
    write_to_cache(CACHE_FOLDER, {'ch1_fname': str(ch1_fname), 'ch1_frames': ch1_frames, 'ch2_fname': str(ch2_fname), 'ch2_frames': ch2_frames, 'results_fname': str(results_fname), 'cell_radius': cell_radius})
    print("reading files")
    ch1 = tifffile.imread(str(ch1_fname))[ch1_slice]
    if ch1.ndim == 3:
        ch1 = ch1.mean(axis=0)
    ch2 = tifffile.imread(str(ch2_fname))[ch2_slice]
    if ch2.ndim == 3:
        ch2 = ch2.mean(axis=0)
    ch1, ch2 = _normalize_arrays(ch1, ch2)
    print("finished reading tiffs")
    im = cv2.addWeighted(ch1, 0.5, ch2, 0.5, 0)
    new_fname = ch1_fname.parent / ('combined_' + ch1_fname.stem + '_' + ch2_fname.stem + '.tif')
    roi_fname = str(new_fname.parent / ('only_roi_' + ch1_fname.stem + '_' + ch2_fname.stem + '.tif'))
    new_fname = str(new_fname)
    tifffile.imwrite(new_fname, np.stack([ch1, ch2]))
    fig = plot_cells_and_traces.show_side_by_side([im], [results_fname], None, cell_radius)
    plot_cells_and_traces.draw_rois_over_cells(im, cell_radius, results_file=results_fname, roi_fname=roi_fname)
    # ch1 -= ch1.min()
    # ch2 -= ch2.min()
    vmin1, vmax1 = ch1.min() * 1.1, ch1.max() * 0.9
    vmin2, vmax2 = ch2.min() * 1.1, ch2.max() * 0.9
    fig.axes[0].images.pop()
    fig.axes[0].imshow(ch1, cmap=cc.cm.kg, vmin=vmin1, vmax=vmax1)
    fig.axes[0].imshow(ch2, cmap=cc.cm.kr, alpha=0.55, vmin=vmin2, vmax=vmax2)
    fig.axes[0].set_title('Ch1 is green, Ch2 is red')
    fig.canvas.set_window_title(f"{new_fname}")
    plt.show(block=False)
    fig.savefig(str(pathlib.Path(new_fname).with_suffix('.pdf')), transparent=True, dpi=300)
    return new_fname


if __name__ == '__main__':
    data = read_from_cache(CACHE_FOLDER)
    with event_loop():
        gui = overlay_channels_and_show_traces.Gui(show=True)
        if data:
            gui.ch1_fname = data['ch1_fname']
            gui.ch2_fname = data['ch2_fname']
            gui.results_fname = data['results_fname']
            gui.cell_radius = data['cell_radius']
            gui.ch1_frames = data['ch1_frames']
            gui.ch2_frames = data['ch2_frames']
        gui.called.connect(lambda x: gui.set_widget("Message:", str(x), position=-1))



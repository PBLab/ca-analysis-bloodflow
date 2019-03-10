import matplotlib
matplotlib.use('TkAgg')
from roipoly import roipoly
import tifffile
import numpy as np
from scipy.io import loadmat
from matplotlib.gridspec import GridSpec
from typing import List, Dict, Tuple
from collections import namedtuple
from tkinter import filedialog
from tkinter import *
from PIL import Image
from h5py import File
from os.path import splitext
import random
import matplotlib.pyplot as plt
import h5py
from trace_converter import ConversionMethod, RawTraceConverter
from analysis_gui import AnalysisGui
from analog_trace import AnalogTraceAnalyzer
import pandas as pd
import xarray as xr
import json
from calcium_trace_analysis import CalciumAnalyzer
from calcium_over_time import AnalyzeCalciumOverTime
from pathlib import Path
from guis_for_analysis import PrelimGui, verify_prelim_gui_inputs
import sys


def batch_process(foldername, close_figs=True):
    """
    Run analysis on all files in folder
    :return:
    """
    # foldername = Path(r'X:\David\THY_1_GCaMP_BEFOREAFTER_TAC_290517')
    # all_files = foldername.rglob('*DAY*EXP_STIM*FOV*.tif')
    all_files = Path(foldername).rglob('*vessels_only*.mat')
    do_calcium = False
    for file in all_files:
        if 'oldana' in str(file) or 'Oldana' in str(file):
            pass
        else:
            print(f"Starting {str(file)}...")
            main(filename=str(file), save_file=False, run_gui=False, do_calcium=do_calcium)
            if close_figs:
                plt.close('all')

def main(filename=None, save_file=False, run_gui=True,
         do_calcium=True, do_vessels=True) -> Dict:
    """ Analyze calcium traces and compare them to vessel diameter """

    # Parameters
    colors = [f"C{idx}" for idx in range(10)] * 3 # use default matplotlib colormap
    return_vals = {}
    if run_gui:
        gui = AnalysisGui()
        gui.root.mainloop()
    else:
        pass

    if filename is None:
        root1 = Tk()
        root1.withdraw()
        filename = filedialog.askopenfilename(title="Choose a stack for cell ROIs",
                                              filetypes=[("Tiff Stack", "*.tif"), ("HDF5 Stack", "*.h5"),
                                                         ("HDF5 Stack", "*.hdf5")],
                                              initialdir=r'/data/David/')
    fpath = str(Path(filename).parent / Path(f"vessel_neurons_analysis_{Path(filename).name[:-4]}"))

    try:
        ca_analysis = gui.ca_analysis.get()
    except (UnboundLocalError, NameError):
        ca_analysis = do_calcium
    if ca_analysis:
        img_neuron, time_vec, fluo_trace, rois = determine_manual_or_auto(filename=Path(filename),
                                                                          time_per_frame=1/float(gui.frame_rate.get()),
                                                                          num_of_rois=int(gui.num_of_rois.get()),
                                                                          colors=colors,
                                                                          num_of_channels=gui.num_of_chans.get(),
                                                                          channel_to_keep=gui.chan_of_neurons.get())
        return_vals['fluo_trace'] = fluo_trace
        return_vals['time_vec'] = time_vec
        return_vals['img_neuron'] = img_neuron
        return_vals['cells_filename'] = Path(filename).name[:-4]
        if type(rois[0]) == roipoly:  # extract coords
            rois = [np.array([roi.allxpoints, roi.allypoints]) for roi in rois]
        return_vals['rois'] = rois

    try:
        bloodflow_analysis = gui.bloodflow_analysis.get()
    except (UnboundLocalError, NameError):
        bloodflow_analysis = do_vessels
    if bloodflow_analysis:
        basename = Path(filename).name[:-4]
        patrick_mat = filename
        # patrick = Path(filename).parent.glob("*" + basename + "*vessels*.mat")  # COMMENTED BY HAGAI FOR RAT ANALYSIS
        # try:
        #     patrick_mat = str(next(patrick))
        # except StopIteration:
        #     warnings.warn(f"File {filename} has no Patrick .mat file.")
        # else:
        if True:
            struct_name = "mv_mpP"
            vessel_lines, diameter_data, img_vessels = import_andy_and_plot(filename=patrick_mat,
                                                                            struct_name=struct_name,
                                                                            colors=colors)
            return_vals['diameter_data'] = diameter_data
            return_vals['img_vessels'] = img_vessels
            return_vals['vessels_filename'] = Path(patrick_mat).name[:-4]
            return_vals['vessel_lines'] = vessel_lines


            if ca_analysis:
                idx_of_closest_vessel = find_closest_vessel(rois=rois, vessels=vessel_lines)

                plot_neuron_with_vessel(rois=rois, vessels=vessel_lines, closest=idx_of_closest_vessel,
                                        img_vessels=img_vessels, fluo_trace=fluo_trace, time_vec=time_vec,
                                        diameter_data=diameter_data, img_neuron=img_neuron)
                return_vals['idx_of_closest_vessel'] = idx_of_closest_vessel

    if gui.analog_trace.get():
        analog_data_fname = next(Path(filename).parent.glob('*analog.txt'))
        analog_data = pd.read_table(analog_data_fname, header=None,
                                    names=['stimulus', 'run'], index_col=False)
        an_trace = AnalogTraceAnalyzer(filename, analog_data)
        an_trace.run()
        sliced_fluo: xr.DataArray = an_trace * return_vals['fluo_trace']  # Overloaded __mul__
        with open(fpath + 'sliced_fluo_traces.json', 'w') as f:
            json.dump(sliced_fluo.to_dict(), f)

        # Further analysis of sliced calcium traces follows
        analyzed_data = CalciumAnalyzer(sliced_fluo)
        analyzed_data.run_analysis()
    plt.show(block=False)

    if save_file:
        np.savez(fpath + '.npz', **return_vals)
    return return_vals


def determine_manual_or_auto(filename: Path, fps: float,
                            num_of_rois: int, colors: List, num_of_channels: int,
                            channel_to_keep: int):
    """
    Helper function to decide whether to let the user draw the ROIs himself or use an existing .npz file
    :param filename:
    :param time_per_frame:
    :param num_of_rois:
    :param colors:
    :param num_of_channels:
    :param channel_to_keep:
    :return:
    """
    name = splitext(filename.name)[0]
    parent_folder = filename.parent
    try:
        corresponding_npz = next(parent_folder.glob(name + "*results.npz"))
        # corresponding_npz = next(parent_folder.glob("results_onACID_" + name + "*.npz"))
        img_neuron, time_vec, fluo_trace, rois = parse_npz_from_caiman(filename=corresponding_npz,
                                                                       fps=fps)
    except StopIteration:
        img_neuron, time_vec, fluo_trace, rois = draw_rois_and_find_fluo(filename=str(filename),
                                                                         time_per_frame=1/fps,
                                                                         num_of_rois=num_of_rois, colors=colors,
                                                                         num_of_channels=num_of_channels,
                                                                         channel_to_keep=channel_to_keep)
    except:
        raise ValueError("Unknown error.")

    return img_neuron, time_vec, fluo_trace, rois


def parse_npz_from_caiman(filename: Path, fps=15.24):

    # Setup - load file and create figure

    sys.path.append(r'/data/Hagai/Multiscaler/code_for_analysis')
    import caiman_funcs_for_comparison


    full_dict = np.load(str(filename))
    fig = plt.figure()
    r = lambda: random.randint(0, 255)
    colors = [f"C{idx}" for idx in range(10)]

    # Get image and plot it
    img_neuron = full_dict['Cn']
    ax_img = fig.add_subplot(121)
    ax_img.imshow(img_neuron, cmap='gray')
    ax_img.set_axis_off()
    ax_img.set_title("Field of View")

    # Generate ROIs and plot them
    rois = []
    rel_crds = full_dict['crd_good']
    for idx, item in enumerate(rel_crds):
        cur_coor = item['coordinates']
        cur_coor = cur_coor[~np.isnan(cur_coor)].reshape((-1, 2))
        rois.append(item['CoM'])
        ax_img.plot(cur_coor[:, 0], cur_coor[:, 1], colors[idx % 10])
        min_c, max_c = cur_coor[:, 0].max(), cur_coor[:, 1].max()
        ax_img.text(min_c, max_c, str(idx+1), color='w')


    # Plot the fluorescent traces
    ax_fluo = fig.add_subplot(122)

    fluo_trace = caiman_funcs_for_comparison.detrend_df_f_auto(full_dict['A'], full_dict['b'], full_dict['C'],
                                                               full_dict['f'], full_dict['YrA'])  # offline pipeline
    num_of_rois, num_of_slices = fluo_trace.shape[0], fluo_trace.shape[1]
    time_vec = np.arange(start=0, stop=1/fps*(fluo_trace.shape[1]), step=1/fps)
    time_vec = np.tile(time_vec, (num_of_rois, 1))
    converted_trace = RawTraceConverter(conversion_method=ConversionMethod.NONE,
                                        raw_data=fluo_trace).convert()

    ax_fluo.plot(time_vec.T, converted_trace.T)
    ax_fluo.set_xlabel("Time [sec]")
    ax_fluo.set_ylabel("Cell ID")
    ax_fluo.set_yticks(np.arange(num_of_rois) + 0.5)
    ax_fluo.set_yticklabels(np.arange(1, num_of_rois + 1))
    ax_fluo.set_title(r'$\frac{\Delta F}{F} $')
    ax_fluo.spines['top'].set_visible(False)
    ax_fluo.spines['right'].set_visible(False)
    return img_neuron, time_vec, fluo_trace, rois


def draw_rois_and_find_fluo(filename: str, time_per_frame: float,
                            num_of_rois: int, colors: List, num_of_channels: int,
                            channel_to_keep: int):
    """

    :param filename:
    :param time_per_frame: 1/Hz (1/framerate)
    :param num_of_rois: Number of GCaMP-labeled cells to mark
    :param colors:
    :param num_of_channels: How many channels does the stack contain
    :param channel_to_keep: Which channel (1..end) contains the GCaMP data?
    :return:
    """
    print("Reading stack...")
    if filename.endswith(".tif"):
        tif = tifffile.imread(filename)
        data = tif[channel_to_keep-1::num_of_channels, :]
    elif filename.endswith(".h5") or filename.endswith(".hdf5"):
        with File(filename, 'r') as h5file:
            data = h5file['mov'].value  # EP's motion correction output

    # If image isn't symmetric - expand it. Useful for PYSIGHT outputs
    if data.shape[1] != data.shape[2]:
        data = resize_image(data)

    print("Reading complete.")
    num_of_slices = data.shape[0]
    max_time = num_of_slices * time_per_frame  # second
    rois = []
    fluorescent_trace = np.zeros((num_of_rois, num_of_slices))

    # Display the mean image and draw ROIs

    mean_image = np.mean(data, 0)
    for idx in range(num_of_rois):
        fig_rois = plt.figure()
        ax_rois = fig_rois.add_subplot(111)
        ax_rois.imshow(mean_image, cmap='gray')
        rois.append(roipoly(roicolor=colors[idx]))
        plt.show(block=True)

    # Calculate the mean fluo. and draw the cells in a single figure
    print("Calculating mean...")
    fig_cells = plt.figure()
    ax_cells = fig_cells.add_subplot(121)
    ax_cells.set_title("Field of View")
    ax_cells.imshow(mean_image, cmap='gray')
    ax_cells.set_axis_off()

    for idx, roi in enumerate(rois):
        cur_mask = roi.getMask(mean_image)
        fluorescent_trace[idx, :] = np.mean(data[:, cur_mask], axis=-1)
        roi.displayROI()

    con_method = ConversionMethod.DFF  # what to do with the data
    final_fluo = RawTraceConverter(conversion_method=con_method,
                                   raw_data=fluorescent_trace)\
        .convert()

    time_vec = np.linspace(start=0, stop=max_time, num=num_of_slices).reshape((1, num_of_slices))
    time_vec = np.tile(time_vec, (num_of_rois, 1))
    assert time_vec.shape == final_fluo.shape

    # Plot fluorescence results
    ax = fig_cells.add_subplot(122)
    ax.plot(time_vec.T, final_fluo.T)
    ax.set_xlabel("Time [sec]")
    ax.set_ylabel("Cell ID")
    ax.set_yticks(np.arange(num_of_rois) + 0.5)
    ax.set_yticklabels(np.arange(1, num_of_rois + 1))
    ax.set_title(r"$\Delta F / F")

    return mean_image, time_vec, final_fluo, rois


def resize_image(data: np.array) -> np.array:
    """ Change the image size to be symmetric """
    im = Image.fromarray(np.rollaxis(data, 0, 3))
    size = 512, 512
    resized = im.resize(size, Image.BICUBIC)
    data = resized.getdata()
    return data

def import_andy_and_plot(filename: str, struct_name: str, colors: List):
    """
    Import the output of the VesselDiameter.m Matlab script and draw it
    :param filename:
    :param struct_name: inside struct
    :return:
    """
    print(filename)
    try:
        andy = loadmat(filename)
    except (NotImplementedError, ValueError):
        with h5py.File(filename, driver='core') as f:
            andy = f[struct_name]
            img = np.array(f[andy['first_frame'][0][0]]).T
            num_of_vessels = andy['Vessel'].shape[0]
            fig_vessels = plt.figure()
            fig_vessels.suptitle(f"Vessel Diameter Over Time\n{filename}")
            gs1 = GridSpec(num_of_vessels, 2)

            ax_ves = plt.subplot(gs1[:, 0])
            ax_ves.imshow(img, cmap='gray')

            ax_dia = []
            diameter_data = []
            vessel_lines = []
            Line = namedtuple('Line', ('x1', 'x2', 'y1', 'y2'))
            for idx in range(num_of_vessels):
                ax_dia.append(plt.subplot(gs1[idx, 1]))  # Axes of GridSpec
                diameter_data.append(np.array(f[andy['Vessel'][idx, 0]]['diameter']))
                line_x1, line_x2 = f[andy['Vessel'][idx, 0]]['vessel_line/position/xy'][0, :]
                line_y1, line_y2 = f[andy['Vessel'][idx, 0]]['vessel_line/position/xy'][1, :]
                vessel_lines.append(Line(x1=line_x1, x2=line_x2, y1=line_y1, y2=line_y2))

            for idx in range(num_of_vessels):
                ax_dia[idx].plot(np.arange(diameter_data[idx].shape[0]), diameter_data[idx], color=colors[idx])
                ax_ves.plot([vessel_lines[idx].x1, vessel_lines[idx].x2],
                            [vessel_lines[idx].y1, vessel_lines[idx].y2],
                            color=colors[idx])

    else:
        img = andy[struct_name][0][0]['first_frame']
        num_of_vessels = andy[struct_name].shape[1]
        fig_vessels = plt.figure()
        fig_vessels.suptitle("Vessel Diameter Over Time")
        gs1 = GridSpec(num_of_vessels, 2)

        ax_ves = plt.subplot(gs1[:, 0])
        ax_ves.imshow(img, cmap='gray')

        ax_dia = []
        diameter_data = []
        vessel_lines = []
        Line = namedtuple('Line', ('x1', 'x2', 'y1', 'y2'))
        for idx in range(num_of_vessels):
            ax_dia.append(plt.subplot(gs1[idx, 1]))  # Axes of GridSpec
            diameter_data.append(andy[struct_name][0, idx]['Vessel']['diameter'][0][0])
            line_x1, line_x2 = andy[struct_name][0, idx]['Vessel']['vessel_line'][0][0][0][0][0][0][0][0][:, 0]
            line_y1, line_y2 = andy[struct_name][0, idx]['Vessel']['vessel_line'][0][0][0][0][0][0][0][0][:, 1]
            vessel_lines.append(Line(x1=line_x1, x2=line_x2, y1=line_y1, y2=line_y2))

        for idx in range(num_of_vessels):
            ax_dia[idx].plot(np.arange(diameter_data[idx].shape[1]), diameter_data[idx].T, color=colors[idx])
            ax_ves.plot([vessel_lines[idx].x1, vessel_lines[idx].x2],
                        [vessel_lines[idx].y1, vessel_lines[idx].y2],
                        color=colors[idx])

    plt.savefig(filename[:-4] + '.png')
    return vessel_lines, diameter_data, img


def find_closest_vessel(rois: List[np.ndarray], vessels: List) -> np.array:
    """ For a list of ROIs, find the index of the nearest blood vessel """

    com_vessels = np.zeros((len(vessels), 2))
    for idx, vessel in enumerate(vessels):
        com_vessels[idx, :] = np.mean((vessel.x1, vessel.x2)), np.mean((vessel.y1, vessel.y2))

    idx_of_closest_vessel = np.zeros((len(rois)), dtype=int)
    for idx, roi in enumerate(rois):
        com_x, com_y = roi[0], roi[1]  # center of ROI
        helper_array = np.tile(np.array([com_x, com_y]).reshape((1, 2)), (com_vessels.shape[0], 1))
        dist = np.sqrt(np.sum((helper_array - com_vessels) ** 2, 1))
        idx_of_closest_vessel[idx] = np.argmin(dist)

    return idx_of_closest_vessel


def plot_neuron_with_vessel(rois: List[np.ndarray], vessels: List, closest: np.array, img_neuron: np.array,
                            fluo_trace: np.array, time_vec: np.array, diameter_data: List, img_vessels: np.array):
    """ Plot them together """

    # Inits
    fig_comp = plt.figure()
    fig_comp.suptitle("Neurons With Their Closest Vessel")
    gs2 = GridSpec(len(rois) * 2, 3)
    r = lambda: random.randint(0, 255)
    # colors = ['#%02X%02X%02X' % (r(), r(), r()) for idx in range(200)]
    colors = [f"C{idx}" for idx in range(10)]

    # Show image with contours on one side
    ax_img = plt.subplot(gs2[:, 0])
    ax_img.imshow(img_vessels, cmap='gray')
    ax_img.imshow(img_neuron, cmap='cool', alpha=0.5)
    ax_img.set_axis_off()
    plotted_vessels_idx = []
    drawn_cells_idx = []
    for idx, vessel in enumerate(vessels):
        try:
            cur_cells, = np.where(closest == idx)
        except ValueError:  # no result
            continue
        else:
            if len(cur_cells) > 0:
                ax_img.plot([vessel.x1, vessel.x2],
                            [vessel.y1, vessel.y2],
                            color=colors[idx])
                plotted_vessels_idx.append(idx)
                drawn_cells_idx.append(cur_cells[0])
                # Draw the first cell
                circ = plt.Circle((rois[cur_cells[0]][0], rois[cur_cells[0]][1]), edgecolor=colors[idx], fill=False)
                ax_img.add_artist(circ)

    # Go through rois and plot two traces
    ax_neurons = []
    ax_vessels = []
    counter = 0
    for idx in drawn_cells_idx:
        ax_neurons.append(plt.subplot(gs2[idx * 2, 1]))
        ax_vessels.append(plt.subplot(gs2[idx * 2 + 1, 1]))
        ax_neurons[-1].plot(time_vec[idx, :], fluo_trace[idx, :], color=colors[plotted_vessels_idx[counter]])
        cur_closest_vessel = closest[idx]
        if cur_closest_vessel in plotted_vessels_idx:
            closest_vessel = np.squeeze(diameter_data[cur_closest_vessel])
            ax_vessels[-1].plot(time_vec[idx, :], closest_vessel, color=colors[plotted_vessels_idx[counter]])
            counter += 1
            corr = np.correlate(closest_vessel, fluo_trace[idx, :], mode='same') / fluo_trace[idx, :].shape[0]
            corr_ax = plt.subplot(gs2[idx*2 : idx*2+2, 2])
            corr_ax.plot(corr)
            corr_ax.set_xticks([])
            corr_ax.set_yticks([])
            corr_ax.set_xlabel("Closest vessel's diameter")
            corr_ax.set_ylabel("Fluorescence")


def display_data(fname):
    """
    Create a plot of vessels with their diameter data and the respective neuron trace.
    :param fname:
    :return:
    """
    # Inits
    data = np.load(fname)
    fig_comp = plt.figure()
    fig_comp.suptitle("Neurons With Their Closest Vessel")
    gs2 = GridSpec(len(data['rois']) * 2, 2)
    r = lambda: random.randint(0, 255)
    # colors = ['#%02X%02X%02X' % (r(), r(), r()) for idx in range(200)]
    colors = [f"C{idx}" for idx in range(10)]

    # Show image with contours on one side
    ax_img = plt.subplot(gs2[:, 0])
    ax_img.imshow(data['img_vessels'], cmap='gray')
    ax_img.imshow(data['img_neuron'], cmap='cool', alpha=0.5)
    ax_img.set_axis_off()
    for idx, vessel in enumerate(data['vessel_lines']):
        ax_img.plot([vessel.x1, vessel.x2],
                    [vessel.y1, vessel.y2],
                    color=colors[idx])
        try:
            cur_cells, = np.where(data['idx_of_closest_vessel'] == idx)
        except ValueError:  # no result
            continue
        for loc in cur_cells:
            circ = plt.Circle((data['rois'][loc][0], data['rois'][loc][1]), edgecolor=colors[idx], fill=False)
            ax_img.add_artist(circ)


def run():
    pre_gui = PrelimGui()
    pre_gui.root.mainloop()
    verify_prelim_gui_inputs(pre_gui)

    if pre_gui.calcium_over_days.get():
        # start folder choosing
        foldername = filedialog.askdirectory(title='Choose the base directory containing all files',
                                             initialdir=r'/data/David')
        result = AnalyzeCalciumOverTime(foldername, pre_gui.analog.get())
    else:
        filename = filedialog.askopenfilename(title="Choose a stack for analysis",
                                              filetypes=[("Tiff Stack", "*.tif"), ("HDF5 Stack", "*.h5"),
                                                         ("HDF5 Stack", "*.hdf5")],
                                              initialdir=r'/data/David/')

        result = Analyze(filename, pre_gui.calcium_after_caiman.get(),
                         pre_gui.vasculature.get(),
                         pre_gui.analog.get())



if __name__ == '__main__':
    # vals = main(save_file=False, do_vessels=False)
    # foldername = Path(r'X:\David\rat_#919_280917')
    # batch_process(foldername, close_figs=True)
    # Iterate over cells
    #INDICES_FROM_SI = [0, 7, 14, 22, 32, 37, 38, 43]
    # display_data(fname=r'X:\David\THY_1_GCaMP_BEFOREAFTER_TAC_290517\029_HYPER_DAY_0__EXP_STIM\vessel_neurons_analysis_029_HYPER_DAY_0__EXP_STIM__FOV_2_00001.npz')
    result = AnalyzeCalciumOverTime(Path(r'/data/David/THY_1_GCaMP_BEFOREAFTER_TAC_290517')).run_batch_of_timepoint()
    # res = AnalyzeCalciumOverTime(Path(r'X:\David\602_new_baseline_imaging_201217')).read_dataarrays_over_time('spont')
    # res = AnalyzeCalciumOverTime(Path(r'X:\David\602_new_baseline_imaging_201217')).\
    # calc_df_f_over_time(Path(r'X:\David\602_new_baseline_imaging_201217\602_HYPER_DAY_0__EXP_STIM__FOV_3_00001.tif'))

import matplotlib.pyplot as plt
from roipoly import roipoly
import tifffile
import numpy as np
from scipy.io import loadmat
from matplotlib.gridspec import GridSpec
from typing import List, Dict
from collections import namedtuple
from tkinter import ttk
from tkinter import filedialog
from tkinter import *
from PIL import Image
from pathlib import Path
from h5py import File
from os.path import splitext
import random


def main() -> Dict:
    """ Analyze calcium traces and compare them to vessel diameter """

    # Parameters
    colors = [f"C{idx}" for idx in range(10)]  # use default matplotlib colormap

    # Main GUI
    root = Tk()
    root.title("Choose what you'd like to analyze")
    frame = ttk.Frame(root)
    frame.pack()
    style = ttk.Style()
    style.theme_use("clam")

    ca_analysis = BooleanVar(value=True)
    bloodflow_analysis = BooleanVar(value=True)
    frame_rate = StringVar(value="30.03")  # Hz
    num_of_rois = StringVar(value="1")
    num_of_chans = IntVar(value=2)
    chan_of_neurons = IntVar(value=1)

    check_cells = ttk.Checkbutton(frame, text="Analyze calcium?", variable=ca_analysis)
    check_cells.pack()
    check_bloodflow = ttk.Checkbutton(frame, text="Analyze bloodflow?", variable=bloodflow_analysis)
    check_bloodflow.pack()
    label_rois = ttk.Label(frame, text="Number of cell ROIs: ")
    label_rois.pack()
    rois_entry = ttk.Entry(frame, textvariable=num_of_rois)
    rois_entry.pack()
    label_time_per_frame = ttk.Label(frame, text="Frame rate [Hz]: ")
    label_time_per_frame.pack()
    time_per_frame_entry = ttk.Entry(frame, textvariable=frame_rate)
    time_per_frame_entry.pack()
    label_num_of_chans = ttk.Label(frame, text="Number of channels: ")
    label_num_of_chans.pack()
    num_of_chans_entry = ttk.Entry(frame, textvariable=num_of_chans)
    num_of_chans_entry.pack()
    label_chan_of_neurons = ttk.Label(frame, text="Channel of neurons: ")
    label_chan_of_neurons.pack()
    chan_of_neurons_entry = ttk.Entry(frame, textvariable=chan_of_neurons)
    chan_of_neurons_entry.pack()

    root.mainloop()

    return_vals = {}

    if ca_analysis.get():
        root1 = Tk()
        root1.withdraw()
        filename = filedialog.askopenfilename(title="Choose a stack for cell ROIs",
                                              filetypes=[("Tiff stack", "*.tif"), ("HDF5 stack", "*.h5")])
        img_neuron, time_vec, fluo_trace, rois = determine_manual_or_auto(filename=Path(filename),
                                                                          time_per_frame=1/float(frame_rate.get()),
                                                                          num_of_rois=int(num_of_rois.get()),
                                                                          colors=colors,
                                                                          num_of_channels=num_of_chans.get(),
                                                                          channel_to_keep=chan_of_neurons.get())
        return_vals['fluo_trace'] = fluo_trace
        return_vals['time_vec'] = time_vec
        return_vals['img_neuron'] = img_neuron
        return_vals['cells_filename'] = Path(filename).name[:-4]
        if type(rois[0]) == roipoly:  # extract CoM
            rois = [np.array([np.mean(roi.allxpoints), np.mean(roi.allypoints)]) for roi in rois]
        return_vals['rois'] = rois

    if bloodflow_analysis.get():
        basename = Path(filename).name[:-4]
        patrick = Path(filename).parent.glob("*" + basename + "*vessels*.mat")
        for cur_file in patrick:
            patrick_mat = str(cur_file)
        if patrick_mat is None:
            raise UserWarning("Patrick's output .mat file not found in folder.")
        struct_name = "mv_mpP"
        vessel_lines, diameter_data, img_vessels = import_andy_and_plot(filename=patrick_mat,
                                                                        struct_name=struct_name,
                                                                        colors=colors)
        return_vals['diameter_data'] = diameter_data
        return_vals['img_vessels'] = img_vessels
        return_vals['vessels_filename'] = Path(patrick_mat).name[:-4]

        if ca_analysis.get():
            idx_of_closest_vessel = find_closest_vessel(rois=rois, vessels=vessel_lines)

            plot_neuron_with_vessel(rois=rois, vessels=vessel_lines, closest=idx_of_closest_vessel,
                                    img_vessels=img_vessels, fluo_trace=fluo_trace, time_vec=time_vec,
                                    diameter_data=diameter_data, img_neuron=img_neuron)
            return_vals['idx_of_closest_vessel'] = idx_of_closest_vessel

    plt.show(block=False)
    return return_vals


def determine_manual_or_auto(filename: Path, time_per_frame: float,
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
        corresponding_npz = next(parent_folder.glob("*" + name + ".npz"))
        img_neuron, time_vec, fluo_trace, rois = parse_npz_from_caiman(filename=corresponding_npz,
                                                                       time_per_frame=time_per_frame)
    except StopIteration:
        img_neuron, time_vec, fluo_trace, rois = draw_rois_and_find_fluo(filename=filename,
                                                                         time_per_frame=time_per_frame,
                                                                         num_of_rois=num_of_rois, colors=colors,
                                                                         num_of_channels=num_of_channels,
                                                                         channel_to_keep=channel_to_keep)
    except:
        raise ValueError("Unknown error.")

    return img_neuron, time_vec, fluo_trace, rois


def parse_npz_from_caiman(filename: Path, time_per_frame: float):

    # Setup - load file and create figure
    full_dict = np.load(str(filename), encoding='bytes')
    fig = plt.figure()
    r = lambda: random.randint(0, 255)
    colors = ['#%02X%02X%02X' % (r(),r(),r()) for idx in range(100)]

    # Get image and plot it
    img_neuron = full_dict['Cn']
    ax_img = fig.add_subplot(121)
    ax_img.imshow(img_neuron, cmap='gray')
    ax_img.set_axis_off()
    ax_img.set_title("Field of View")

    # Generate ROIs and plot them
    rois = []
    for idx, item in enumerate(full_dict['crd']):
        cur_coor = item[b'coordinates']
        cur_coor = cur_coor[~np.isnan(cur_coor)].reshape((-1, 2))
        ax_img.plot(cur_coor[:, 0], cur_coor[:, 1], colors[idx])
        rois.append(item[b'CoM'])

    # Plot the fluorescent traces
    ax_fluo = fig.add_subplot(122)
    fluo_trace = full_dict['Cf']
    num_of_rois, num_of_slices = fluo_trace.shape[0], fluo_trace.shape[1]
    offset_vec = np.arange(num_of_rois).reshape((num_of_rois, 1))
    offset_vec = np.tile(offset_vec, num_of_slices)
    fps = full_dict['metadata'][0][b'SI.hRoiManager.scanFrameRate']
    time_vec = np.arange(start=0, stop=1/fps*(fluo_trace.shape[1]), step=1/fps)
    time_vec = np.tile(time_vec, (num_of_rois, 1))

    final_trace = compute_final_trace(fluo_trace)
    assert offset_vec.shape == final_trace.shape

    fluorescent_trace_normed_off = final_trace + offset_vec
    assert time_vec.shape == fluorescent_trace_normed_off.shape

    ax_fluo.plot(time_vec.T, fluorescent_trace_normed_off.T)
    ax_fluo.set_xlabel("Time [sec]")
    ax_fluo.set_ylabel("Cell ID")
    ax_fluo.set_yticks(np.arange(num_of_rois) + 0.5)
    ax_fluo.set_yticklabels(np.arange(1, num_of_rois + 1))
    ax_fluo.set_title("Fluorescence trace")
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
    elif filename.endswith(".h5"):
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

    fluorescent_trace_normed = compute_final_trace(fluorescent_trace)

    offset_vec = np.arange(num_of_rois).reshape((num_of_rois, 1))
    offset_vec = np.tile(offset_vec, num_of_slices)
    assert offset_vec.shape == fluorescent_trace_normed.shape

    fluorescent_trace_normed_off = fluorescent_trace_normed + offset_vec
    time_vec = np.linspace(start=0, stop=max_time, num=num_of_slices).reshape((1, num_of_slices))
    time_vec = np.tile(time_vec, (num_of_rois, 1))
    assert time_vec.shape == fluorescent_trace_normed_off.shape

    # Plot fluorescence results
    ax = fig_cells.add_subplot(122)
    ax.plot(time_vec.T, fluorescent_trace_normed_off.T)
    ax.set_xlabel("Time [sec]")
    ax.set_ylabel("Cell ID")
    ax.set_yticks(np.arange(num_of_rois) + 0.5)
    ax.set_yticklabels(np.arange(1, num_of_rois + 1))
    ax.set_title("Fluorescence Trace")

    return mean_image, time_vec, fluorescent_trace_normed, rois


def resize_image(data: np.array) -> np.array:
    """ Change the image size to be symmetric """
    im = Image.fromarray(np.rollaxis(data, 0, 3))
    size = 512, 512
    resized = im.resize(size, Image.BICUBIC)
    data = resized.getdata()
    return data


def compute_final_trace(trace: np.array) -> np.array:
    """
    Take a fluorescent trace and normalize it for display purposes.
    :param trace: Raw trace
    :return: np.array of the normalized trace, a row per ROI.
    """
    num_of_rois = trace.shape[0]
    num_of_slices = trace.shape[1]

    median_f0 = np.mean(trace, 1).reshape((num_of_rois, 1))  # CHANGED FROM MEDIAN TO MEAN
    median_f0 = np.tile(median_f0, num_of_slices)
    assert median_f0.shape == trace.shape
    df = trace - median_f0
    df_over_f = df / median_f0

    return df_over_f


def import_andy_and_plot(filename: str, struct_name: str, colors: List):
    """
    Import the output of the VesselDiameter.m Matlab script and draw it
    :param filename:
    :param struct_name: inside struct
    :return:
    """
    andy = loadmat(filename)
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
    gs2 = GridSpec(len(rois) * 2, 2)
    r = lambda: random.randint(0, 255)
    colors = ['#%02X%02X%02X' % (r(), r(), r()) for idx in range(100)]

    # Show image with contours on one side
    ax_img = plt.subplot(gs2[:, 0])
    ax_img.imshow(img_vessels, cmap='gray')
    ax_img.imshow(img_neuron, cmap='cool', alpha=0.5)
    ax_img.set_axis_off()
    for idx, vessel in enumerate(vessels):
        ax_img.plot([vessel.x1, vessel.x2],
                    [vessel.y1, vessel.y2],
                    color=colors[idx])
        try:
            cur_cells, = np.where(closest == idx)
        except ValueError:  # no result
            continue
        for loc in cur_cells:
            circ = plt.Circle((rois[loc][0], rois[loc][1]), edgecolor=colors[idx], fill=False)
            ax_img.add_artist(circ)

    # Go through rois and plot two traces
    ax_neurons = []
    ax_vessels = []
    for idx in range(len(rois)):
        ax_neurons.append(plt.subplot(gs2[idx * 2, 1]))
        ax_vessels.append(plt.subplot(gs2[idx * 2 + 1, 1]))
        ax_neurons[idx].plot(time_vec[idx, :], fluo_trace[idx, :], color=colors[idx])
        closest_vessel = diameter_data[closest[idx]]
        ax_vessels[idx].plot(np.arange(closest_vessel.shape[1]), closest_vessel.T, color=colors[idx])


if __name__ == '__main__':
    vals = main()
    # np.save(f'vals_{vals["cells_filename"]}.npy', vals)
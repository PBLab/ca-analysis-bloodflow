"""
__author__ = Hagai Har-Gil
"""

from tkinter import ttk
from tkinter import filedialog
from tkinter import *


class AnalysisGui:

    """
    Main GUI for calcium-bloodflow analysis pipeline
    """
    def __init__(self):
        self.root = Tk()
        self.root.title("Choose what you'd like to analyze")
        frame = ttk.Frame(self.root)
        frame.pack()
        style = ttk.Style()
        style.theme_use("clam")

        self.ca_analysis = BooleanVar(value=True)
        self.bloodflow_analysis = BooleanVar(value=False)
        self.frame_rate = StringVar(value="15.24")  # Hz
        self.num_of_rois = StringVar(value="1")
        self.num_of_chans = IntVar(value=1)
        self.chan_of_neurons = IntVar(value=1)
        self.analog_trace = BooleanVar(value=True)

        check_cells = ttk.Checkbutton(frame, text="Analyze calcium?", variable=self.ca_analysis)
        check_cells.pack()

        check_analog = ttk.Checkbutton(frame, text="Contains analog channel?", variable=self.analog_trace)
        check_analog.pack()

        check_bloodflow = ttk.Checkbutton(frame, text="Analyze bloodflow?", variable=self.bloodflow_analysis)
        check_bloodflow.pack()

        label_rois = ttk.Label(frame, text="Number of cell ROIs: ")
        label_rois.pack()
        rois_entry = ttk.Entry(frame, textvariable=self.num_of_rois)
        rois_entry.pack()
        label_time_per_frame = ttk.Label(frame, text="Frame rate [Hz]: ")
        label_time_per_frame.pack()
        time_per_frame_entry = ttk.Entry(frame, textvariable=self.frame_rate)
        time_per_frame_entry.pack()
        label_num_of_chans = ttk.Label(frame, text="Number of channels: ")
        label_num_of_chans.pack()
        num_of_chans_entry = ttk.Entry(frame, textvariable=self.num_of_chans)
        num_of_chans_entry.pack()
        label_chan_of_neurons = ttk.Label(frame, text="Channel of neurons: ")
        label_chan_of_neurons.pack()
        chan_of_neurons_entry = ttk.Entry(frame, textvariable=self.chan_of_neurons)
        chan_of_neurons_entry.pack()

        self.root.wait_window()
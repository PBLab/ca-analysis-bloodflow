"""
__author__ = Hagai Hargil
"""
from tkinter import filedialog
from tkinter import *
from tkinter import ttk
from enum import Enum


class ChosenPipeline(Enum):
    CALCIUM_AFTER_CAIMAN = 'Analyze calcium from single CaImAn-analyzed timelapse'
    CALCIUM_ACROSS_DAYS = 'Analyze calcium across days'
    MATCH_CALCIUM_VESSELS = "Associate calcium activity and vasculature (following Patrick's code)"
    ANALOG = 'Analog data was recorded'

class PrelimGui(object):
    """
    GUI to choose the exact pipeline the user wishes to run
    """
    def __init__(self):
        self.root = Tk()
        self.root.title("Neurovascular Pipeline")
        self.root.rowconfigure(16, weight=1)
        self.root.columnconfigure(16, weight=1)
        main_frame = ttk.Frame(self.root, width=1000, height=1300)
        main_frame.grid(column=0, row=0)
        main_frame['borderwidth'] = 2
        style = ttk.Style()
        style.theme_use('clam')

        self.__title(main_frame)
        self.__selection(main_frame)

        # Define the last quit button and wrap up GUI
        quit_button = ttk.Button(main_frame, text='Next', command=self.root.destroy,
                                 underline=0)
        quit_button.grid(row=13, column=4, sticky='ns')

        self.root.bind("n", self.__dest)
        self.root.bind("<Return>", self.__dest)
        for child in main_frame.winfo_children():
            child.grid_configure(padx=3, pady=2)

        self.root.wait_window()

    def __dest(self, event):
        self.root.destroy()

    def __title(self, frame):
        lbl = ttk.Label(frame, text='Choose analysis modality:')
        lbl.pack()

    def __selection(self, frame):
        self.calcium_after_caiman = StringVar()
        self.calcium_over_days = StringVar()
        self.vasculature = StringVar()
        self.analog = StringVar()

        cb_after = ttk.Checkbutton(frame, text=ChosenPipeline.CALCIUM_AFTER_CAIMAN.value,
                                   variable=self.calcium_after_caiman)
        cb_after.pack()
        cb_over_days = ttk.Checkbutton(frame, text=ChosenPipeline.CALCIUM_ACROSS_DAYS.value,
                                       variable=self.calcium_over_days)
        cb_over_days.pack()
        cb_vasc = ttk.Checkbutton(frame, text=ChosenPipeline.MATCH_CALCIUM_VESSELS.value,
                                  variable=self.vasculature)
        cb_vasc.pack()
        cb_analog = ttk.Checkbutton(frame, text=ChosenPipeline.ANALOG.value,
                                    variable=self.analog)
        cb_analog.pack()


def verify_prelim_gui_inputs(gui):
    """
    Make sure user data is valid
    :param gui:
    :return:
    """
    if gui.vasculature.get() and gui.calcium_over_days.get():
        raise UserWarning('Vasculature analysis not supported with "analysis over days".')

    if gui.calcium_over_days.get() and not gui.calcium_after_caiman.get():
        raise UserWarning("Manual markings and analysis over days not supported currently.")

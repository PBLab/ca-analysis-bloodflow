import attr
from attr.validators import instance_of
import numpy as np
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import tifffile
import os
import itertools
import xarray as xr
from collections import namedtuple
from datetime import datetime
import colorama
colorama.init()
import copy
import warnings

from ca_analysis.analog_trace import AnalogTraceAnalyzer
from ca_analysis.dff_tools import calc_dff, calc_dff_batch, scatter_spikes, plot_mean_vals, display_heatmap


@attr.s(slots=True)
class VascOccParser:
    """
    A class that provides the analysis pipeline for stacks with vascular occluder. Meaning,
    Data acquired in a before-during-after scheme, where "during" is the perturbation done
    to the system, occlusion of an artery in this case. The class needs to know how many frames
    were acquired before the perturbation and how many were acquired during. It also needs
    other metadata, such as the framerate, and the IDs of cells that the CaImAn pipeline
    accidentally labeled as active components. If the data contains analog recordings as well,
    of the mouse's movements and air puffs, they will be integrated into the analysis as well.
    If one of the data channels contains co-labeling with a different, usually morphological,
    fluorophore indicating the cell type, it will be integrated as well.
    """
    foldername = attr.ib(validator=instance_of(str))
    glob = attr.ib(default='*results.npz', validator=instance_of(str))
    fps = attr.ib(default=15.24, validator=instance_of(float))
    frames_before_stim = attr.ib(default=1000)
    len_of_epoch_in_frames = attr.ib(default=1000)
    with_analog = attr.ib(default=False, validator=instance_of(bool))
    with_colabeling = attr.ib(default=False, validator=instance_of(bool))
    serialize = attr.ib(default=True, validator=instance_of(bool))
    dff = attr.ib(init=False)
    frames_after_stim = attr.ib(init=False)
    start_time = attr.ib(init=False)
    timestamps = attr.ib(init=False)
    num_of_channels = attr.ib(init=False)
    sliced_fluo = attr.ib(init=False)
    OccMetadata = attr.ib(init=False)
    data_files = attr.ib(init=False)
    colabel_idx = attr.ib(init=False)

    def __attrs_post_init__(self):
        self.OccMetadata = namedtuple('OccMetadata', ['before', 'during', 'after'])

    def run(self):
        self._find_all_files()
        self._get_params()
        if self.with_analog:
            self.__run_with_analog()  # colabeling check is inside there
            self.dff = self.sliced_fluo.loc['all'].values
        elif self.with_colabeling:
            self.dff = self._load_dff()
            self.colabel_idx = self._load_colabeled_idx()
        else:
            self.dff = calc_dff_batch(self.data_files['caiman'])

    def __run_with_analog(self):
        """ Helper function to run sequentially all needed analysis of dF/F + Analog data """
        list_of_sliced_fluo = []  # we have to compare each file with its analog data, individually
        for idx, row in self.data_files.iterrows():
            dff = calc_dff((row['caiman']))
            analog_data = pd.read_table(row['analog'], header=None,
                                        names=['stimulus', 'run'], index_col=False)
            occ_metadata = self.OccMetadata(self.frames_before_stim, self.len_of_epoch_in_frames,
                                            self.frames_after_stim)
            analog_trace = AnalogTraceAnalyzer(row['caiman'], analog_data, framerate=self.fps,
                                               start_time=self.start_time,
                                               timestamps=self.timestamps,
                                               occluder=True, occ_metadata=occ_metadata)
            analog_trace.run()
            copied_trace = copy.deepcopy(analog_trace)  # for some reason,
            # multiplying the trace by the dff changes analog_trace. To overcome
            # this weird issue we're copying it.
            list_of_sliced_fluo.append(analog_trace * dff)  # overloaded __mul__
        print("Concatenating FOVs into a single data structure...")
        self.sliced_fluo: xr.DataArray = concat_vasc_occ_dataarrays(list_of_sliced_fluo)
        if self.with_colabeling:
            self.colabel_idx = self._load_colabeled_idx()
        if self.serialize:
            print("Writing to disk...")
            self._serialize_results(row['tif'].parent)

    def _serialize_results(self, foldername: pathlib.Path):
        """ Write to disk the generated concatenated DataArray """
        self.sliced_fluo.attrs['fps'] = self.fps
        self.sliced_fluo.attrs['frames_before_occ'] = self.frames_before_stim
        self.sliced_fluo.attrs['frames_during_occ'] = self.len_of_epoch_in_frames
        self.sliced_fluo.attrs['frames_after_occ'] = self.frames_after_stim
        if self.with_colabeling:
            self.sliced_fluo.attrs['colabeled'] = self.colabel_idx

        self.sliced_fluo.to_netcdf(str(foldername / 'vasc_occ_parsed.nc'), mode='w')  # TODO: compress

    def _find_all_files(self):
        """
        Locate all fitting files in the folder - The correpsonding "sibling" files
        for the main TIF recording, like analog data recordings, etc.
        """
        self.data_files = pd.DataFrame([], columns=['caiman', 'tif', 'analog', 'colabeled'])
        folder = pathlib.Path(self.foldername)
        files = folder.rglob(self.glob)
        print("Found the following files:")
        for idx, file in enumerate(files):
            print(file)
            cur_file = os.path.splitext(str(file.name))[0][:-18]  # no "_CHANNEL_X_results"
            try:
                raw_tif = next(file.parent.glob(cur_file + '.tif'))
            except StopIteration:
                print(f"No corresponding Tiff found for file {cur_file}.")
                raw_tif = ''

            try:
                analog_file = next(file.parent.glob(cur_file + '_analog.txt'))  # no
            except StopIteration:
                print(f"No corresponding analog data found for file {cur_file}.")
                analog_file = ''

            try:
                colabeled_file = next(file.parent.glob(cur_file + '*_colabeled*.npy'))
            except StopIteration:
                print(f"No corresponding colabeled channel found for file {cur_file}. Did you run 'batch_colabeled'?")
                colabeled_file = ''

            self.data_files = self.data_files.append(pd.DataFrame([[str(file), raw_tif, analog_file, colabeled_file]],
                                                                  columns=['caiman', 'tif', 'analog', 'colabeled'],
                                                                  index=[idx]))

    def _get_params(self):
        """ Get general stack parameters from the TiffFile object """
        try:
            print("Getting TIF parameters...")
            with tifffile.TiffFile(self.data_files['tif'][0]) as f:
                si_meta = f.scanimage_metadata
                self.fps = si_meta['FrameData']['SI.hRoiManager.scanFrameRate']
                self.num_of_channels = len(si_meta['FrameData']['SI.hChannels.channelsActive'])
                num_of_frames = len(f.pages) // self.num_of_channels
                self.frames_after_stim = num_of_frames - (self.frames_before_stim + self.len_of_epoch_in_frames)
                self.start_time = str(datetime.fromtimestamp(os.path.getmtime(self.data_files['tif'][0])))
                self.timestamps = np.arange(num_of_frames)
                print("Done without errors!")
        except TypeError:
            warnings.warn('Failed to parse ScanImage metadata')
            self.start_time = None
            self.timestamps = None
            self.frames_after_stim = 1000

    def _load_colabeled_idx(self):
        """ Loads the indices of the colabeled cells from all found files """
        colabel_idx = []
        num_of_cells = 0
        for _, row in self.data_files.iterrows():
            cur_data = np.load(row.caiman)['F_dff']
            cur_idx = np.load(row.colabeled)
            cur_idx += num_of_cells
            colabel_idx.append(cur_idx)
            num_of_cells += cur_data.shape[0]

        colabel_idx = np.array(list(itertools.chain.from_iterable(colabel_idx)))
        return colabel_idx

    def _load_dff(self):
        """ Loads the dF/F data from all found files """
        dff = []
        for _, row in self.data_files.iterrows():
            cur_data = np.load(row.caiman)['F_dff']
            dff.append(cur_data)
        dff = np.concatenate(dff)
        return dff

    def _display_analog_traces(self, ax_puff, ax_jux, ax_run, data: AnalogTraceAnalyzer):
        """ Show three Axes of the analog data """
        ax_puff.plot(data.stim_vec)
        ax_puff.invert_yaxis()
        ax_puff.set_ylabel('Direct air puff')
        ax_puff.set_xlabel('')
        ax_puff.set_xticks([])
        ax_jux.plot(data.juxta_vec)
        ax_jux.invert_yaxis()
        ax_jux.set_ylabel('Juxtaposed puff')
        ax_jux.set_xlabel('')
        ax_jux.set_xticks([])
        ax_run.plot(data.run_vec)
        ax_run.invert_yaxis()
        ax_run.set_ylabel('Run times')
        ax_run.set_xlabel('')
        ax_run.set_xticks([])

    def _display_occluder(self, ax, data_length):
        """ Show the occluder timings """
        occluder = np.zeros((data_length))
        occluder[self.frames_before_stim:self.frames_before_stim + self.len_of_epoch_in_frames] = 1
        time = np.arange(data_length) / self.fps
        ax.plot(time, occluder)
        ax.get_xaxis().set_ticks_position('top')

        ax.invert_yaxis()

        ax.set_ylabel('Artery occlusion')
        ax.set_xlabel('')

def concat_vasc_occ_dataarrays(da_list: list):
    """ Take a list of DataArrays and concatenate them together
    while keeping the index integrity """
    new_da_list = []
    num_of_neurons = 0
    crd_time = da_list[0].time.values
    crd_epoch = da_list[0].epoch.values
    for idx, da in enumerate(da_list):
        crd_neuron = np.arange(num_of_neurons, num_of_neurons + len(da.neuron))
        if len(da.time) > len(crd_time):
            crd_time = da.time.values
        reindexed_da = xr.DataArray(data=da.data,
                                    dims=['epoch', 'neuron', 'time'],
                                    coords={'epoch': crd_epoch,
                                            'neuron': crd_neuron,
                                            'time': crd_time},
                                    attrs=da.attrs)
        new_da_list.append(reindexed_da)        
        num_of_neurons += len(da.neuron)    
    
    return xr.concat(new_da_list, dim='neuron')


if __name__ == '__main__':
    folder = '/data/David/Vascular occluder_ALL/vip_td_gcamp_270818_muscle_only'
    glob = r'f*results.npz'
    assert pathlib.Path(folder).exists()
    frames_before_stim = 2000
    len_of_epoch_in_frames = 2000
    fps = 58.2
    with_analog = True
    with_colabeling = True
    display_each_fov = False
    serialize = True
    vasc = VascOccParser(foldername=folder, glob=glob,
                         frames_before_stim=frames_before_stim,
                         len_of_epoch_in_frames=len_of_epoch_in_frames,
                         fps=fps,
                         with_analog=with_analog,
                         with_colabeling=with_colabeling,
                         serialize=serialize)
    vasc.run()
    plt.show(block=True)

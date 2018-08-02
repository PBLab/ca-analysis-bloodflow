"""
__author__ = Hagai Har-Gil
"""
import attr
from attr.validators import instance_of
import tifffile
import h5py
import numpy as np
from roipoly import roipoly
import matplotlib.pyplot as plt
from dff_calc.df_f_calculation import DffCalculator


@attr.s
class ManualRoiDrawing:
    """
    Calculate the raw fluorescence and dF/F of manually-drawn ROIs on a stack
    :param fname: data timelapse - .tif or .hdf5 (string)
    :param num_rois: number of ROIs to draw (integer)
    :param fps: frame rate (float)
    :param colors: list of colors to use
    :param scale: Distance between each cell trace corresponds to scale % of dF/F change. Default is 1, i.e. 100% dF/F
    """
    fname = attr.ib(validator=instance_of(str))
    num_rois = attr.ib(validator=instance_of(int))
    fps = attr.ib(default=7.68, validator=instance_of(float))
    colors = attr.ib(default=[f'C{idx}' for idx in range(10)])
    scale = attr.ib(default=1., validator=instance_of(float))
    data = attr.ib(init=False)
    rois = attr.ib(init=False)
    agg_data = attr.ib(init=False)
    raw_traces = attr.ib(init=False)
    dff = attr.ib(init=False)

    def run(self):
        self._load_data()
        self._offset_data()
        self._draw_rois()
        self._get_trace_from_masks()
        self.dff = DffCalculator(self.raw_traces, fps=self.fps).calc()
        self._display_roi_with_trace()
        return self.dff

    def _load_data(self):
        """ Load different data formats into self.data """
        print(f"Loading {self.fname}...")
        if self.fname.endswith('.tif'):
            self.data = tifffile.imread(self.fname)
        elif self.fname.endswith('.h5') or self.fname.endswith('.hdf5'):
            with h5py.File(self.fname) as f:
                self.data = np.array(f['/Full Stack/Channel 1'])
        else:
            raise UserWarning(f'File type of {self.fname} not supported.')

    def _offset_data(self):
        """ Subtract the offset from the raw data """
        dt = self.data.dtype
        if (dt is np.uint32) or (dt is np.int32):
            self.data = self.data.astype(np.int64)

        if (dt is np.float32) or (dt is np.float16):
            self.data = self.data.astype(np.float64)

        self.data -= self.data.min()

    def _draw_rois(self):
        """
        Draw the ROIPoly on the data
        :return:
        """
        self.rois = []
        self.agg_data = self.data.sum(axis=0)
        for idx in range(self.num_rois):
            fig_rois, ax_rois = plt.subplots()
            ax_rois.imshow(self.agg_data, cmap='gray')
            for roi in self.rois:
                roi.displayROI()
            self.rois.append(roipoly(roicolor=self.colors[idx]))
            plt.show(block=True)

    def _get_trace_from_masks(self):
        """ Calculate the mean fluorescent trace in each ROI """
        self.raw_traces = np.zeros((self.num_rois, self.data.shape[0]))

        for idx, roi in enumerate(self.rois):
            cur_mask = roi.getMask(self.agg_data)
            self.raw_traces[idx, :] = np.mean(self.data[:, cur_mask], axis=-1)

    def _display_roi_with_trace(self):
        """ Show the image of the average stack overlaid with the ROIs, and the dF/F values """
        num_of_frames = self.dff.shape[1]
        fig_cells, (ax_cells, ax_trace) = plt.subplots(1, 2)
        plt.sca(ax_cells)
        ax_cells.set_title("Field of View")
        ax_cells.imshow(self.agg_data, cmap='gray')
        for roi in self.rois:
            roi.displayROI()
        ax_cells.set_axis_off()
        max_time = num_of_frames / self.fps

        plt.sca(ax_trace)
        time_vec = np.arange(start=0, stop=max_time, step=1/self.fps)[:num_of_frames].reshape((1, num_of_frames))
        time_vec = np.tile(time_vec, (self.num_rois, 1))
        assert time_vec.shape == self.dff.shape

        # Plot fluorescence results
        trace_locs = np.linspace(start=0, stop=self.num_rois*self.scale, num=self.num_rois, endpoint=False)
        ax_trace.plot(time_vec.T, self.dff.T + trace_locs[np.newaxis, :], linewidth=0.3)
        ax_trace.set_xlabel("Time [sec]")
        ax_trace.set_ylabel("Cell ID")
        ax_trace.set_yticks(trace_locs + 0.2)
        ax_trace.set_yticklabels(np.arange(1, self.num_rois + 1))
        ax_trace.set_title(fr"$\Delta F / F$, scale={self.scale}")

        ax_trace.spines['top'].set_visible(False)
        ax_trace.spines['right'].set_visible(False)

        plt.savefig(f"rois_and_dff_{self.fname.split('/')[-1][:-4]}.pdf", transparent=True)


if __name__ == '__main__':
    manual_rois = []
    dffs = []
    fps = [30.03, 15.24]
    files = [r'/data/Hagai/Multiscaler/27-9-17/For article/Calcium/si_500_frames.tif',
             r'/data/Hagai/Multiscaler/27-9-17/For article/Calcium/pysight_500_frames.tif']
    for file, fr in zip(files, fps):
        manroi = ManualRoiDrawing(fname=file, num_rois=10, fps=fr, scale=0.2)
        dffs.append(manroi.run())
        manual_rois.append(manroi)

    plt.show(block=False)
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# from IPython import get_ipython

# %%
import pathlib
from pprint import pprint as print


# %%
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# %%
import matplotlib.pyplot as plt
import numpy as np
# get_ipython().run_line_magic('matplotlib', 'inline')


# %%
import calcium_bflow_analysis.dff_analysis_and_plotting.plot_cells_and_traces as plot_cells_and_traces
import calcium_bflow_analysis.single_fov_analysis as single_fov
from calcium_bflow_analysis.analog_trace import AnalogAcquisitionType
from calcium_bflow_analysis.fluo_metadata import FluoMetadata


# %%
foldername = pathlib.Path('/data/Hagai/WFA_InVivo/')
tifs = [
    foldername / 'z300_7.tif',
    foldername / 'z400_4.tif',
    foldername / 'z200_4.tif',
]
ch1s = [
    foldername / 'z300_7_CHANNEL_1.tif',
    foldername / 'z400_4_CHANNEL_1.tif',
    foldername / 'z200_4_CHANNEL_1.tif',
]
results = [
    foldername / 'z300_7_CHANNEL_1_results.npz',
    foldername / 'z400_4_CHANNEL_1_results.npz',
    foldername / 'z200_4_CHANNEL_1_results.npz',
]
analogs =  [
    foldername / 'z300_7_analog.txt',
    foldername / 'z400_4_analog.txt',
    foldername / 'z200_4_analog.txt',
]


# %%
# fig = plot_cells_and_traces.show_side_by_side(ch1s, results)


# %%
# fig.savefig(foldername / 'side_by_side.png', transparent=True, dpi=300)


# %%
filenum = 1
meta = FluoMetadata(tifs[filenum], fps=58, num_of_channels=2)
meta.get_metadata()

# %%
fov = single_fov.SingleFovParser(analogs[filenum], results[filenum], metadata=meta, analog=AnalogAcquisitionType.TREADMILL, summarize_in_plot=True)


# %%
fov.parse()


# %%




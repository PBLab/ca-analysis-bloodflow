"""
This file loads and processes the data returned from the Sequencer website.
It assumes that it was ran with TNSE enabled, so each results file actually consists
of four pairs of files - the data and the indices. We generally only use the indices
files since the data was already loaded into memory once.

The main function here is "create_data_dictionary", which returns a dict with all of
the data that was downloaded from the sequencer.
"""
import pathlib
from collections import namedtuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr

from calcium_bflow_analysis.sequencer.label_sequencer_indices import get_non_nan_data_and_labels


SeqResult = namedtuple('SeqResult', ['data', 'fnames', 'mouseids'])


def _get_key_name(fname):
    """An internal function used to determine the type of the current dataset.

    'original' - untouched data (not returned by this function)
    'seq' - sequencer result
    'tsne_10' - TNSE with p=10
    'tsne_50' - TNSE with p=50
    'tsne_100' - TNSE with p=100
    """
    file = str(fname)
    if 'tsne' not in file:
        return 'seq'

    if 'p10.txt' in file:
        return 'tsne_10'

    if 'p50.txt' in file:
        return 'tsne_50'

    if 'p100.txt' in file:
        return 'tsne_100'


def get_data_type_and_indices(fname):
    """For a given filename find the appropriate key for the data dictionary,
    and load the indices of it."""
    key = _get_key_name(fname)
    indices = np.loadtxt(fname, delimiter=',', dtype=np.uint32)
    return key, indices


def create_data_dictionary(folder):
    """Returns a dictionary, with each key being a data type, and the values
    are a namedtuple that contains all needed information of these traces."""
    files = folder.glob('sorting_indexes*')
    datadict = {}
    data, fname_labels, mouseid_labels = get_non_nan_data_and_labels(next(folder.glob('data_of_day_1.nc')))
    datadict['original'] = SeqResult(data, fname_labels, mouseid_labels)
    for file in files:
        key, indices = get_data_type_and_indices(file)
        new_data = data[indices, :]
        new_fnames = fname_labels[indices]
        new_mids = mouseid_labels[indices]
        datadict[key] = SeqResult(new_data, new_fnames, new_mids)
    return datadict


if __name__ == '__main__':
    folder = pathlib.Path('/data/Amit_QNAP/Calcium_FXS')
    datadict = create_data_dictionary(folder)

"""
This file contains functions that map the new indices Sequencer returned to the original
ones that were given when xarray parsed them.
"""
import pathlib
from typing import Tuple

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt


def read_xr_and_concat(fname: pathlib.Path):
    """Reads the given filename and concatenates it into a single file.

    Assumes that the filename is an xarray file which was made by parsing
    many calcium analysis results.
    """
    data = xr.open_dataset(fname).dff
    return np.vstack(data)


def find_non_nan_rows(data: np.ndarray):
    """Return the indices of the non-nan rows of the data."""
    nans = np.isnan(data).any(axis=1)
    return ~nans


def make_filename_labels(fname: pathlib.Path):
    """Finds the filename for each of the dF/F traces and returns a vector
    with the full length of dF/F cells which labels which cell originated
    from which file."""
    data = xr.open_dataset(fname)
    return np.repeat(data.fname, len(data.neuron)).values


def make_mouse_id_labels(fname: pathlib.Path):
    """Finds the mouse ID for each of the dF/F traces and returns a vector
    with the full length of dF/F cells which labels which cell originated
    from which mouse."""
    data = xr.open_dataset(fname)
    return np.repeat(data.mouse_id, len(data.neuron)).values


def get_non_nan_data_and_labels(fname) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A pipeline function which wraps all existing functionality in this
    file by reading the data and returning the non-NaN rows of it, as well
    as the labels for each of these rows."""
    data = read_xr_and_concat(fname)
    non_nan_rows = find_non_nan_rows(data)
    data = data[non_nan_rows, :]
    fname_labels = make_filename_labels(fname)[non_nan_rows]
    mouseid_labels = make_mouse_id_labels(fname)[non_nan_rows]
    return data, fname_labels, mouseid_labels


if __name__ == '__main__':
    foldername = pathlib.Path('/data/Amit_QNAP/Calcium_FXS/')
    fname = foldername / 'data_of_day_1.nc'
    assert fname.exists()
    data, fname_labels, mouseid_labels = get_non_nan_data_and_labels(fname)

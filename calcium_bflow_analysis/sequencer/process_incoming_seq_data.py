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
from typing import MutableMapping

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import seaborn as sns

from calcium_bflow_analysis.sequencer.label_sequencer_indices import (
    get_non_nan_data_and_labels,
)


SeqResult = namedtuple("SeqResult", ["data", "fnames", "mouseids", "indices"])


def _get_key_name(fname):
    """An internal function used to determine the type of the current dataset.

    'original' - untouched data (not returned by this function)
    'seq' - sequencer result
    'tsne_10' - TNSE with p=10
    'tsne_50' - TNSE with p=50
    'tsne_100' - TNSE with p=100
    """
    file = str(fname)
    if "tsne" not in file:
        return "seq"

    if "p10.txt" in file:
        return "tsne_10"

    if "p50.txt" in file:
        return "tsne_50"

    if "p100.txt" in file:
        return "tsne_100"


def get_data_type_and_indices(fname):
    """For a given filename find the appropriate key for the data dictionary,
    and load the indices of it."""
    key = _get_key_name(fname)
    indices = np.loadtxt(fname, delimiter=",", dtype=np.uint32)
    return key, indices


def create_data_dictionary(folder):
    """Returns a dictionary, with each key being a data type, and the values
    are a namedtuple that contains all needed information of these traces."""
    files = folder.glob("sorting_indexes*")
    datadict = {}
    data, fname_labels, mouseid_labels = get_non_nan_data_and_labels(
        next(folder.glob("data_of_day_1.nc"))
    )
    datadict["original"] = SeqResult(
        data, fname_labels, mouseid_labels, np.arange(len(data))
    )
    for file in files:
        key, indices = get_data_type_and_indices(file)
        new_data = data[indices, :]
        new_fnames = fname_labels[indices]
        new_mids = mouseid_labels[indices]
        datadict[key] = SeqResult(new_data, new_fnames, new_mids, indices)
    return datadict


def _concat_datadict_into_longform(datadict: MutableMapping[str, SeqResult]):
    """Creates a long-form DF which is suitable for plotting the changes in ordering
    of FOVs and mouse IDs after the sequencer."""
    df = pd.DataFrame([], columns=["Origin", "FOV", "MouseID", "Index"])
    ordered_fovs = datadict["original"].fnames
    ordered_ids = datadict["original"].mouseids
    unique_fovs = np.unique(ordered_fovs)
    unique_ids = np.unique(ordered_ids)
    fov_map = {k: v for k, v in zip(unique_fovs, np.arange(len(unique_fovs)))}
    ids_map = {k: v for k, v in zip(unique_ids, np.arange(len(unique_ids)))}
    mapping = (
        pd.DataFrame({"fov": ordered_fovs, "mouse": ordered_ids})
        .assign(fov_num=lambda x: x["fov"].apply(lambda x: fov_map[x]))
        .assign(mouse_num=lambda x: x["mouse"].apply(lambda x: ids_map[x]))
    )
    all_data = []
    for key, dataseq in datadict.items():
        cur_df = df.copy()
        cur_df["FOV"] = mapping['fov_num'].iloc[dataseq.indices]
        cur_df["MouseID"] = mapping['mouse_num'].iloc[dataseq.indices]
        cur_df["Origin"] = key
        cur_df["Index"] = dataseq.indices
        all_data.append(cur_df)

    return pd.concat(all_data)


def _make_fig(ax, title, fovs, mids):
    """Plot the new data ordering for both the FOVs and Mouse IDs."""
    ax_mids = ax.twinx()
    xax = np.arange(len(fovs))
    ax.scatter(xax, fovs, marker='o', c='C0', s=0.5)
    ax_mids.scatter(xax, mids, marker='x', c='C1', s=0.5)
    ax_mids.set_ylabel('Mouse ID')
    ax.set_title(title)
    ax.set_xlabel('Cell Index')


def plot_new_ordering(datadict: MutableMapping[str, SeqResult]):
    """Shows the new ordering of the rows based on their filenames and mouse IDs."""
    fig, axes = plt.subplots(1, 5, figsize=(48, 16), sharex=True, sharey=True)
    for ax, (name, data_seq) in zip(axes, datadict.items()):
        _make_fig(ax=ax, title=name, fovs=data_seq.fnames, mids=data_seq.mouseids)
    axes[0].set_ylabel('FOV')
    axes[0].set_yticklabels([])
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    folder = pathlib.Path("/data/Amit_QNAP/Calcium_FXS")
    datadict = create_data_dictionary(folder)
    fig = plot_new_ordering(datadict)
    plt.show()
    fig.savefig(folder / 'sequencer_output' / 'changes_in_ordering.pdf', transparent=True, dpi=300)

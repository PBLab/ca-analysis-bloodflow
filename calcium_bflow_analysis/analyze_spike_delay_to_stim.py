from calcium_bflow_analysis.dff_analysis_and_plotting import dff_analysis
import pathlib
from calcium_bflow_analysis.single_fov_analysis import filter_da
from typing import Iterator, Tuple

import numpy as np
import xarray as xr
import scipy.spatial.distance


def iter_over_mouse_and_fname(dataset: xr.Dataset) -> Iterator[xr.Dataset]:
    """Construct an iterator over each filename in every mouse in the given
    xarray Dataset.

    This method is useful when working with stimulus data, for example, since
    each filename had its own stimulus timing, so you want to do things at a
    per-filename basis rather than a per-mouse one.

    Parameters
    ----------
    dataset : xr.Dataset
        A "data_of_day_X.nc" dataset resulting from CalciumReview analysis

    Returns
    -------
    ds : Generator[xr.Dataset]
        The dataset containing only the relevant data
    """
    for mouse_id, ds in dataset.groupby('mouse_id'):
        for fname in ds.fname.values:
            print(f"Mouse {mouse_id}; file {fname}")
            yield ds.sel(fname=fname)


def get_dff_spikes_stim(dataset: xr.Dataset, epoch: str, fps: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Helper function to get - per dataset - the dF/F, spikes, and stimulus
    arrays."""
    dff = filter_da(dataset, epoch)
    if len(dff) == 0:
        raise ValueError
    spikes = dff_analysis.locate_spikes_scipy(dff, fps)
    stim = dataset['epoch_times'].sel(epoch='stim').values
    return dff, spikes, stim


def get_shortest_delay_between_stim_and_spike(spike_times: np.ndarray, stim_start_indices: np.ndarray) -> np.ndarray:
    """Find the closest stim to each spike.

    Each spike will receive an index of the stim that it was closest to. This
    index is actually the start time of the stimulus as computed elsewhere.

    Parameters
    ----------
    spike_times : np.ndarray
        Index of the starting spike time, i.e. the column in its spikes matrix

    stim_start_indices : np.ndarray
        Index of the stimulus starting time

    Returns
    -------
    shortest_delay_from_stim_per_spike : np.ndarray
        1D array with the same length as teh spike times, corresponding to the
        stimulus index per each of the spikes
    """
    shortest_delay_from_stim_per_spike = scipy.spatial.distance.cdist(
        np.atleast_2d(spike_times).T,
        np.atleast_2d(stim_start_indices).T
    ).min(axis=1).astype(np.int64)
    assert len(shortest_delay_from_stim_per_spike) == len(spike_times)
    return shortest_delay_from_stim_per_spike


def filter_spikes_that_occurred_out_of_bounds(shortest_delay_from_stim_per_spike: np.ndarray, spikes: np.ndarray, bounds: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Discards spikes that occurred too early or too late relative to their
    closest stimulus.

    For some applications it's desired to only look at a subset of the spikes
    in the experiment. This function discards spikes that occurred in such
    undesireable times, for example spikes that occurred 5 seconds or more
    after a stimulus, which might mean they're less relevant when coding that
    specific stimulus.

    Parameters
    ----------
    shortest_delay_from_stim_per_spike : np.ndarray
        The result of "get_shortest_delay_between_stim_and_spike"

    spikes : np.ndarray
        The spikes matrix, 1 where a spike occurred and 0 otherwise

    bounds : Tuple[int, int]
        Starting and ending allowed indices of the spikes

    Returns
    -------
    shortest_delay_from_stim_per_spike : np.ndarray
        The same array as the input, but with the irrelevant spikes removed

    new_spike_matrix : np.ndarray
        A matrix with the same shape as the original "spikes" input, but having
        the out-of-bounds spikes removed
    """
    spikes_that_occurred_before_min_delay = np.where(
        shortest_delay_from_stim_per_spike <= bounds[0]
    )[0]
    spikes_that_occurred_after_max_delay = np.where(
        shortest_delay_from_stim_per_spike >= bounds[1]
    )[0]
    out_of_bounds_spikes = np.union1d(spikes_that_occurred_before_min_delay, spikes_that_occurred_after_max_delay)
    spike_rows, spike_times = np.where(spikes)
    spike_times = np.delete(
        spike_times,
        out_of_bounds_spikes,
    )
    shortest_delay_from_stim_per_spike = np.delete(
        shortest_delay_from_stim_per_spike,
        out_of_bounds_spikes
    )
    assert len(shortest_delay_from_stim_per_spike) == len(spike_times)
    spike_rows = np.delete(spike_rows, out_of_bounds_spikes)
    new_spike_matrix = np.zeros_like(spikes)
    new_spike_matrix[spike_rows, spike_times] = 1
    return shortest_delay_from_stim_per_spike, new_spike_matrix


def assign_spikes_to_closest_stim(spikes: np.ndarray, stim: np.ndarray, bounds: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Each spike receives a stimulus index which corresponds to its closest
    partner.

    This calculation may be constrained using the bounds parameter, which can
    filter out spikes that happened too soon or too late relative to a stimulus

    Parameters
    ----------
    spikes: np.ndarray
        A cell x time matrix with 1 wherever a spike occurred

    stim : np.ndarray
        1D array with True wherever a stim happened

    bounds : Tuple[int, int]
        Start and end delays, in number of steps (not seconds)

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Closest stimulus per spikes, and a modified spikes array

    Raises
    ------
    ValueError
        If no spikes or no stimuli occurred in this dataset
    """
    stim_start_indices = np.where(np.concatenate([np.diff(stim), [False]]))[0][::2]
    if len(stim_start_indices) == 0 or spikes.sum() == 0:
        raise ValueError
    shortest_delay_from_stim_per_spike = get_shortest_delay_between_stim_and_spike(
        np.where(spikes)[1],
        stim_start_indices
    )
    shortest_delay_from_stim_per_spike, spikes = filter_spikes_that_occurred_out_of_bounds(
        shortest_delay_from_stim_per_spike,
        spikes,
        bounds,
    )
    return shortest_delay_from_stim_per_spike, spikes


def get_traces_around_spikes(spikes: np.ndarray, dff: np.ndarray, fps: float, win_length: float) -> np.ndarray:
    """For each spike get the corresponding dF/F trace.

    This function is used when we wish to observe the dF/F behavior of a cell
    during its spikes, i.e. when we need more than the straight-forward timing
    of a spike. This can be thought of "bloating" the area around a spike.

    Parameters
    ----------
    spikes : np.ndarray
        cell x time matrix with 1 wherever a spike occurred
    dff : np.ndarray
        cell x time
    fps : float
        Frame rate
    win_length : float
        Length of the bloating window in seconds. The first second comes
        "before" the spike, and the rest will come after

    Returns
    -------
    bloated : np.ndarray
        A spike x time array, with its number of columns equalling fps *
        win_length cells
    """
    one_sec = int(fps)
    remaining_window_length = int((win_length * fps) - fps)
    assert remaining_window_length > 0
    bloated = np.zeros((int(spikes.sum()), one_sec + remaining_window_length), dtype=np.float32)
    bloated = populated_bloated_with_dff(spikes, dff, bloated, one_sec, remaining_window_length)
    return bloated


def populated_bloated_with_dff(spikes: np.ndarray, dff: np.ndarray, bloated: np.ndarray, before_spike, after_spike) -> np.ndarray:
    """Populates an array with the dF/F data of each spike.

    Parameters
    ----------
    spikes : np.ndarray
        cell x time matrix with 1 wherever a spike occurred
    dff : np.ndarray
        cell x time matrix with the dF/F data
    bloated : np.ndarray
        A pre-allocated zeroed array with a shape of num_spikes x win_length
    before_spike, after_spike : int
        Number of cells to capture before and after each spike

    Returns
    -------
    bloated : np.ndarray
        The populated array with the dF/F data
    """
    rows, columns = np.where(spikes > 0)
    for spike_id, (row, column) in enumerate(zip(rows, columns)):
        try:
            bloated[spike_id] = dff[row, column - before_spike:column + after_spike]
        except ValueError:  # spike was too close to the edge of the recording
            continue
    return bloated


def delay_spikes_as_dff(bloated: np.ndarray, shortest_delay_from_stim_per_spike: np.ndarray, total_window: float, fps: float) -> Tuple[np.ndarray, np.ndarray]:
    """Place each spike's dF/F in an array at the proper delay relative to a
    stimulus.

    Each line in bloated is the dF/F trace of a spike, and this function will
    place this spike in a wider array that also takes into account the timing
    of this spike relative to the closest stimulus.

    Parameters
    ----------
    bloated : np.ndarray
        spikes x time dF/F array
    shortest_delay_from_stim_per_spike : np.ndarray
        1D array with the number of cells this spike was delays by relative to
        the stimulus
    total_window : float
        Total window size to display, in seconds
    fps : float
        Frame rate

    Returns
    -------
    dff_spikes : np.ndarray
        An array with the same shape as bloated but with the spikes shifted by
        some amount
    delays : np.ndarray
        The delay of each spike in seconds
    """
    length_of_a_single_spike = bloated.shape[1]
    num_of_spikes = len(bloated)
    dff_spikes = np.zeros((num_of_spikes, int(total_window * fps)), dtype=bloated.dtype)
    delays = np.zeros(num_of_spikes, dtype=np.float32)
    for spike_idx, (bloated_spike, delay) in enumerate(zip(bloated, shortest_delay_from_stim_per_spike)):
        dff_spikes[spike_idx, delay:(delay + length_of_a_single_spike)] = bloated_spike[:]
        delays[spike_idx] = np.float32(delay / fps)

    return dff_spikes, delays

if __name__ == '__main__':
    fname = pathlib.Path('/data/Amit_QNAP/Calcium_FXS/data_of_day_1.nc')
    data = xr.open_dataset(fname)
    fps = data.fps
    MIN_DELAY_BETWEEN_STIM_AND_SPIKE = int(1.8 * fps)
    MAX_DELAY_BETWEEN_STIM_AND_SPIKE = int(8 * fps)
    fxs = ['518', '609', '614', '647', '648', '650']
    wt = ['293', '595', '596', '615', '640', '674']
    epoch = 'all'
    all_fxs_spikes = []
    all_fxs_delays = []
    all_wt_spikes = []
    all_wt_delays = []
    per_fname_ds_iter = iter_over_mouse_and_fname(data)
    for ds in per_fname_ds_iter:
        fps = ds.fps
        all_ids = ds.mouse_id.values
        try:
            len(all_ids)
        except TypeError:
            mouse_id = all_ids
        else:
            mouse_id = all_ids[0]
        dff, spikes, stim = get_dff_spikes_stim(ds, epoch, fps)
        try:
            shortest_delay_from_stim_per_spike, new_spike_matrix = assign_spikes_to_closest_stim(
                spikes,
                stim,
                (MIN_DELAY_BETWEEN_STIM_AND_SPIKE, MAX_DELAY_BETWEEN_STIM_AND_SPIKE),
            )
        except ValueError:
            print(f"No spikes occurred in {ds.fname.values}")
            continue
        bloated = get_traces_around_spikes(new_spike_matrix, dff, fps, 3)
        assert len(bloated) == len(shortest_delay_from_stim_per_spike)

        spikes, delays = delay_spikes_as_dff(bloated, shortest_delay_from_stim_per_spike, 13, fps)
        if mouse_id in fxs:
            all_fxs_spikes.append(spikes)
            all_fxs_delays.append(delays)
        elif mouse_id in wt:
            all_wt_spikes.append(spikes)
            all_wt_delays.append(delays)

    all_fxs_spikes = np.concatenate(all_fxs_spikes)
    all_fxs_delays = np.concatenate(all_fxs_delays)
    all_wt_spikes = np.concatenate(all_wt_spikes)
    all_wt_delays = np.concatenate(all_wt_delays)

from typing import Tuple, List, MutableMapping, Union

import numpy as np
import xarray as xr


def dff_dataset_init(
    data_vars: MutableMapping[str, Tuple[List[str], np.ndarray]],
    coords: MutableMapping[str, np.ndarray],
    attrs: MutableMapping[str, Union[int, float]],
) -> xr.Dataset:
    """
    The only place where the datasets holding the sliced dF/F data
    are allowed to be created. It sanitizes the inputs and always
    returns a valid xr.Dataset instance. If something in the inputs
    is invalid, it raises a ValueError. The allowed or mandatory
    inputs are listed at the start of this function.
    """
    allowed_datavars_keys = set(("dff", "epoch_times"))
    if set(data_vars.keys()) != allowed_datavars_keys:
        raise ValueError(
            f"data_vars keys were invalid. Expected {allowed_datavars_keys}, received {data_vars.keys()}."
        )

    allowed_coords_keys = set(
        ("neuron", "time", "epoch", "fov", "mouse_id", "condition", "day", "fname")
    )
    if set(coords.keys()) != allowed_coords_keys:
        raise ValueError(
            f"coords keys were invalid. Expected {allowed_coords_keys}, received {coords.keys()}."
        )

    mandatory_attrs_keys = set(("fps", "stim_window"))
    if not mandatory_attrs_keys.issubset(set(attrs.keys())):
        raise ValueError(
            f"The attrs variable must contain the following keys: {mandatory_attrs_keys}, however it contained the following: {attrs.keys()}"
        )

    return xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)

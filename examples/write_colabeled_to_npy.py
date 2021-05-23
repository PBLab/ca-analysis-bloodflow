import pathlib
from magicgui import magicgui
import numpy as np


@magicgui(
    call_button="Save",
    persist=True,
    main_window=True,
    fname={"label": "Original file name"},
    cell_numbers={"label": "Cell numbers"},
)
def serialize_colabeled_indices(fname: pathlib.Path, cell_numbers: str):
    """Writes to disk the colabeled cells in a parseable format.

    This small utility is usually used in conjuction with the 'overlay' GUI to
    write to disk the indices of cells which are considered colabeled. The
    resulting file will have a '*_colabeled.npy' suffix to it.

    Parameters
    ----------
    fname : pathlib.Path
        Filename for the original data. Leave as is, this tool will add the
        needed suffix.
    cell_numbers : str
        A comma-separated list of integers representing the cell numbers as
        given in the overlay tool or in other CaImAn-related tools.
    """
    new_fname = modify_fname_for_colabeled(fname)
    cells = np.asarray(cell_numbers.strip(' ,').split(','), dtype=np.int32)
    np.save(new_fname, cells)
        

def modify_fname_for_colabeled(fname: pathlib.Path) -> pathlib.Path:
    old_name = fname.stem
    old_name += "_colabeled.npy"
    new_fname = fname.with_name(old_name)
    return new_fname


if __name__ == '__main__':
    serialize_colabeled_indices.show(run=True)

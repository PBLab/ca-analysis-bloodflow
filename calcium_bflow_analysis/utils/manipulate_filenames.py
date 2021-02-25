import pathlib


def add_id(fname: pathlib.Path, id_: str):
    old_name = fname.name
    new_name = f"mouse_{id_}_" + old_name
    new_fname = fname.with_name(new_name)
    return fname.rename(new_fname)


def replace_day(fname: pathlib.Path, old: str, new: str):
    old_name = fname.name
    new_name = old_name.replace(old, new)
    new_fname = fname.with_name(new_name)
    return fname.rename(new_fname)


if __name__ == '__main__':
    foldername = pathlib.Path('/data/David/TAC_survivors/D_751_all_after_caiman/D-751-baseline')
    glob = 'D_751*_R_*'
    for file in foldername.rglob(glob):
        new_fname = replace_day(file, 'baseline', 'DAY_0')
        new_fname = replace_day(new_fname, '_R_', '_RH_')
        # add_id(new_fname, '208')

import importlib

from dj_tables import *


def drop_all():
    """
    Removes all manual tables
    """
    ExpParams().drop_quick()
    # ExpParams().drop_quick()
    # ComputedParams().drop_quick()
    # CaimanResults().drop_quick()
    # ManualReview().drop_quick()

def pop_exxpp():
    par = ExppParams()
    par.insert1((1, '1111-11-11', 'a'))


def populate_exp_params():
    params = ExpParams()
    exp_id = 0
    date = "2019-06-11"
    experimenter = "Hagai"
    mouse_number = "629"
    line_injection = "GCaMP6"
    gcamp_type = "Fast"
    experiment_type = "TAC"
    foldername = "/data/"
    glob = "*.tif"
    condition_reg = ""
    num_of_channels = "1"
    calcium_channel = "1"
    lines = 512
    columns = 512
    z_planes = 0
    fr = 58.23
    magnification = 1.0
    bidirectional = "true"
    slow_scan_coef = 1.0
    objective_lens = "x10"
    scan_freq = 7929.0
    cell_radius_x = 5
    cell_radius_y = 5
    cell_radius_z = 0
    cells_per_patch = 2
    params.insert1(
        (
            exp_id,
            date,
            experimenter,
            mouse_number,
            line_injection,
            gcamp_type,
            experiment_type,
            foldername,
            glob,
            condition_reg,
            num_of_channels,
            calcium_channel,
            lines,
            columns,
            z_planes,
            fr,
            magnification,
            bidirectional,
            slow_scan_coef,
            objective_lens,
            scan_freq,
            cell_radius_x,
            cell_radius_y,
            cell_radius_z,
            cells_per_patch,
        )
    )


if __name__ == "__main__":
    drop_all()
    import dj_tables

    dj_tables = importlib.reload(dj_tables)
    # populate_exp_params()
    pop_exxpp()

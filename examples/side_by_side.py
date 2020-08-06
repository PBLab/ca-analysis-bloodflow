import pathlib

from magicgui import magicgui, event_loop

from calcium_bflow_analysis.dff_analysis_and_plotting import plot_cells_and_traces

# linux only
CACHE_FOLDER = pathlib.Path.home() / pathlib.Path('.cache/ca_analysis_bloodflow')


@magicgui(call_button="Show", layout="form")
def overlay_channels_and_show_traces(ch1_fname: str = ".tif", ch2_fname: str = ".tif", results_fname: str = "*.npz", cell_radius: int = 6):
    ch1_fname = pathlib.Path(ch1_fname)
    if not ch1_fname.exists():
        return "Channel 1 path doesn't exist"
    ch2_fname = pathlib.Path(ch2_fname)
    if not ch1_fname.exists():
        return "Channel 2 path doesn't exist"
    results_fname = pathlib.Path(results_fname)
    if not results_fname.exists():
        return "Results path doesn't exist".

    write_to_cache(CACHE_FOLDER, {'ch1_fname': ch1_fname, 'ch2_fname': ch2_fname, 'results_fname': results_fname, 'cell_radius': cell_radius}
    plot_cells_and_traces.show_side_by_side(tiff_files, result_files, None, cell_radius)



if __name__ == '__main__':

    overlay_channels_and_show_traces()


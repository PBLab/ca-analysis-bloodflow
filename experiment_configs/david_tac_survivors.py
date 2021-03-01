
    home = Path("/data/David/")
    folder = Path(r"TAC_survivors")
    results_folder = home / folder
    assert results_folder.exists()
    globstr = "*.tif"
    folder_and_files = {home / folder: globstr}
    analog_type = AnalogAcquisitionType.TREADMILL
    filefinder = FileFinder(
        results_folder=results_folder,
        folder_globs=folder_and_files,
        analog=analog_type,
        with_colabeled=False,
    )
    files_table = filefinder.find_files()
    regex = {
        "cond_reg": r"_mag_3_(\w)_",
        "id_reg": r"^D_(\d+)_",
        "fov_reg": r"FOV_(\d)_",
        "day_reg": r"DAY_(\d+)_"
    }
    res = CalciumAnalysisOverTime(
        files_table=files_table,
        serialize=True,
        folder_globs=folder_and_files,
        analog=analog_type,
        regex=regex,
    )
    # res.run_batch_of_timepoints(results_folder)
    res.generate_ds_per_day(results_folder, '*.nc', recursive=True)

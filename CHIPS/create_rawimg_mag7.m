data_dir = '/data/Hagai/flow_david';
chips_rootdir = '/data/MatlabCode/PBLabToolkit/CalciumDataAnalysis/python-ca-analysis-bloodflow/CHIPS/';
fname = 'fov_new_mag_7_4p25_px_915ms_line_00001.tif';

% Specify the image dimensions
imgSize = [256, 256, 1, 200];
fovSize = 680;  % um
pixelSize = fovSize / imgSize(1);
lineTime = 0.915;  % ms
pixelTime = (lineTime / imgSize(1)) * 1000;  % us

% Specify some data about the image acquisition
acq = struct('isBiDi', true, 'lineTime', lineTime, 'zoom', 7, ...
    'nLinesPerFrameOrig', imgSize(1), 'nPixelsPerLineOrig', imgSize(2), ...
    'frameRate', 4.25, 'nChannels', 1, 'discardFlybackLine', false, ...
    'pixelSize', pixelSize, 'pixelTime', pixelTime);

% Specify the full path to the raw image object
fnRID003 = fullfile(data_dir, fname);

% Specify the channels relevant for this raw image
channels = struct('blood_plasma', 1);

fnCalibration = fullfile(chips_rootdir, 'x20_calib_256px.mat');
calibration = CalibrationPixelSize.load(fnCalibration);

% Create the RawImgDummy object without any interaction
rid003 = RawImgDummy(fnRID003, channels, calibration, acq);
data_dir = '/data/Hagai/Occluder_data_david';
chips_rootdir = '/data/MatlabCode/PBLabToolkit/CalciumDataAnalysis/python-ca-analysis-bloodflow/CHIPS/';
fname = 'fov1_mag_3_256px_4p25Hz_bi_915us_00001.tif';

% Specify the image dimensions
imgSize = [256, 256, 1, 75];
fovSize = 680;  % um
lineTime = 0.915;  % ms
zoom = 3;
frameRate = 4.25;
nChannels = 1;
discardFlybackLine = false;
isBiDi = true;

pixelSize = fovSize / imgSize(1);
pixelTime = (lineTime / imgSize(1)) * 1000;  % us


% Specify some data about the image acquisition
acq = struct('isBiDi', isBiDi, 'lineTime', lineTime, 'zoom', zoom, ...
    'nLinesPerFrameOrig', imgSize(1), 'nPixelsPerLineOrig', imgSize(2), ...
    'frameRate', frameRate, 'nChannels', nChannels, ...
    'discardFlybackLine', discardFlybackLine, ...
    'pixelSize', pixelSize, 'pixelTime', pixelTime);

% Specify the full path to the raw image object
fnRID003 = fullfile(data_dir, fname);

% Specify the channels relevant for this raw image
channels = struct('blood_plasma', 1);

fnCalibration = fullfile(chips_rootdir, 'x20_calib_256px.mat');
calibration = CalibrationPixelSize.load(fnCalibration);

% Create the RawImgDummy object without any interaction
rid = RawImgDummy(fnRID003, channels, calibration, acq);

fs = FrameScan('test', rid);
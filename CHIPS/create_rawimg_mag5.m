data_dir = '/data/Hagai/Occluder_data_david';
chips_rootdir = '/data/MatlabCode/PBLabToolkit/CalciumDataAnalysis/python-ca-analysis-bloodflow/CHIPS/';
fname = 'AVG_fov7_mag_5_512Px_5Hz_uni_00001.tif #2.tif';

% Specify the image dimensions
imgSize = [512, 512, 1, 2000];
fovSize = 950;  % um
lineTime = 0.1265;  % ms
zoom = 5;
frameRate = 5.08;
nChannels = 1;
discardFlybackLine = false;
isBiDi = false;

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

fnCalibration = fullfile(chips_rootdir, 'MOM_x25_calib_512px.mat');
calibration = CalibrationPixelSize.load(fnCalibration);

% Create the RawImgDummy object without any interaction
rid = RawImgDummy(fnRID003, channels, calibration, acq);

fs = FrameScan('mom_unidir_5hz_512px', rid);
fs.process();
fs.plot();
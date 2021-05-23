chips_rootdir = '/data/MatlabCode/PBLabToolkit/CalciumDataAnalysis/python-ca-analysis-bloodflow/CHIPS/';
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

% Specify the channels relevant for this raw image
channels = struct('blood_plasma', 1);

% Load the CalibrationPixelSize object
fnCalibration = fullfile(chips_rootdir, 'x20_calib_256px.mat');
calibration = CalibrationPixelSize.load(fnCalibration);

% Create the Metadata object without any interaction
md002 = Metadata(imgSize, acq, channels, calibration);
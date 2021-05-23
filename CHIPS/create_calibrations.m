%% Generate the x20 @ 256 px calibration object
zoom = 1:20;
fovSize = 950;  % um
pxSize = (1./zoom) .* (fovSize / 256);
imgSize = 512;  % px
objective = 'x25';
date = '2019_07_07';
name = 'MOM_x25_512px';
person = 'David';
funRaw = @CalibrationPixelSize.funRawHyperbola;
cal001 = CalibrationPixelSize(zoom, pxSize, imgSize, objective, date, ...
name, person, funRaw);
cal001.plot()

cal001.save('/data/MatlabCode/PBLabToolkit/CalciumDataAnalysis/python-ca-analysis-bloodflow/CHIPS/MOM_x25_calib_512px.mat');
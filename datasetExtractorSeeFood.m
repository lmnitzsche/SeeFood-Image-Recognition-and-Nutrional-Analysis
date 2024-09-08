fprintf("Downloading Example Food Image data set (77 MB)... ")
filename = matlab.internal.examples.downloadSupportFile('nnet', ...
    'data/ExampleFoodImageDataset.zip');
fprintf("Done.\n")

filepath = fileparts(filename);
dataFolder = fullfile(filepath,'ExampleFoodImageDataset');
unzip(filename,dataFolder);
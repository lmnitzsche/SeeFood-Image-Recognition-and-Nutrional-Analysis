% Logan Nitzsche, Tom O'Connell, Sumitra Shrestha, Caleb Sutton,
% RGB
% See Food
% Digital Image Processing - SIUE

% Dataset directory and categories
datasetDir = 'ExampleFoodImageDataset';
categories = {'sushi', 'sashimi', 'pizza', 'hot_dog', 'hamburger', ...
              'greek_salad', 'french_fries', 'caprese_salad', 'caesar_salad'};

% Parameters for enhancement
scaling_factor = 10;
laplacian_kernal = [0 -1 0; -1 4 -1; 0 -1 0];

% Loop through each category
for i = 1:length(categories)
    category = categories{i};
    categoryDir = fullfile(datasetDir, category);
    
    % Get all image files in the category directory
    imageFiles = dir(fullfile(categoryDir, '*.jpg')); % Assumes .jpg images
    if isempty(imageFiles)
        fprintf('No images found in %s\n', categoryDir);
        continue;
    end
    
    % Ensure output directories exist
    scaledDir = fullfile('scaled', category);
    sharpenedDir = fullfile('sharpened', category);
    wieneredDir = fullfile('wienered', category);
    if ~exist(scaledDir, 'dir'), mkdir(scaledDir); end
    if ~exist(sharpenedDir, 'dir'), mkdir(sharpenedDir); end
    if ~exist(wieneredDir, 'dir'), mkdir(wieneredDir); end

    % Process each image in the category
    for j = 1:length(imageFiles)
        % Read the image
        imageFile = imageFiles(j).name;
        imgPath = fullfile(categoryDir, imageFile);
        img = imread(imgPath);

        % Convert to uint16
        img = im2uint16(img);

        % --SCALE--
        scaled_image = scale_image(img, scaling_factor);

        % --SHARPEN--
        sharpened_image = sharpen_image(img, laplacian_kernal);

        % --WIENER FILTER--
        wiener_filtered_image = wiener_filter(img, 1); % Replace IMG with the blurred version and '1' with sigma value

        % Save processed images
        [~, originalTitle, ~] = fileparts(imageFile);
        scaledFilename = fullfile(scaledDir, [originalTitle '_scaled_factor_' num2str(scaling_factor) '.jpg']);
        sharpenedFilename = fullfile(sharpenedDir, [originalTitle '_sharpened.jpg']);
        wienerFilename = fullfile(wieneredDir, [originalTitle '_wiener_filtered.jpg']);
        
        imwrite(im2uint8(mat2gray(scaled_image)), scaledFilename);
        imwrite(im2uint8(mat2gray(sharpened_image)), sharpenedFilename);
        imwrite(im2uint8(mat2gray(wiener_filtered_image)), wienerFilename);

        % Optionally display progress
        fprintf('Processed %s in %s\n', imageFile, category);
    end
end

% All images processed
fprintf('All images in %s processed.\n', datasetDir);

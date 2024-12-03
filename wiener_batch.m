% Logan Nitzsche
% RGB
% See Food
% Digital Image Processing/Computer Vision - SIUE

% Dataset directory and specific categories
datasetDir = 'base';
categories = {'sushi', 'sashimi', 'pizza'};

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
    
    % Ensure output directory exists
    weinernewDir = fullfile('weinernew', category);
    if ~exist(weinernewDir, 'dir'), mkdir(weinernewDir); end

    % Process each image in the category
    for j = 1:length(imageFiles)
        % Read the image
        imageFile = imageFiles(j).name;
        imgPath = fullfile(categoryDir, imageFile);
        img = imread(imgPath);

        % Convert to uint16
        img = im2uint16(img);

        % --WIENER FILTER--
        wiener_filtered_image = wiener_filter(img, 3); % Apply Wiener filter with sigma = 3

        % Save processed image
        [~, originalTitle, ~] = fileparts(imageFile);
        weinerFilename = fullfile(weinernewDir, [originalTitle '_wiener_filtered_sig3.jpg']);
        imwrite(im2uint8(mat2gray(wiener_filtered_image)), weinerFilename);

        % Optionally display progress
        fprintf('Processed %s in %s\n', imageFile, category);
    end
end

% All images processed
fprintf('All images in %s processed.\n', datasetDir);

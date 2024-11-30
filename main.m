% Logan Nitzsche, Tom O'Connell, Sumitra Shrestha, Caleb Sutton,
% RGB
% See Food
% Digital Image Processing - SIUE

% Input image
img = imread('cricket.png');

% Convert to uint16 
img = im2uint16(img);

% Parameters for enhancement
scaling_factor = 10;
laplacian_kernal = [0 -1 0; -1 4 -1; 0 -1 0];

% --SCALE--
scaled_image = scale_image(img, scaling_factor);

% --SHARPEN--
sharpened_image = sharpen_image(img, laplacian_kernal);

% --WIENER FILTER--
wiener_filtered_image = wiener_filter(img, 1); % Replace IMG with the blurred version of the image and replace '1' with the sigma value that the blurred image uses

% --Gather image metrics here for experiments--
calculateEME(img, scaled_image);
calculateEME(img, wiener_filtered_image);

% Display original, scaled, sharpened, wienered images 
figure;
subplot(1, 4, 1), imshow(img, []), title('Original Image (Grayscale)');
subplot(1, 4, 2), imshow(scaled_image, []), title(['Scaled Image (Factor: ' num2str(scaling_factor) ')']);
subplot(1, 4, 3), imshow(sharpened_image, []), title('Sharpened Image');
subplot(1, 4, 4), imshow(wiener_filtered_image, []), title('Wiener Filter Image');

% Ensure directories exist or create them
if ~exist('scaled', 'dir')
    mkdir('scaled');
end
if ~exist('sharpened', 'dir')
    mkdir('sharpened');
end
if ~exist('wienered', 'dir')
    mkdir('wienered');
end

% Get the original filename without extension
[~, originalTitle, ~] = fileparts('cricket.png');

% Save the processed images in respective folders
scaledFilename = fullfile('scaled', [originalTitle '_scaled_factor_' num2str(scaling_factor) '.jpg']);
imwrite(im2uint8(mat2gray(scaled_image)), scaledFilename); % Save scaled image

sharpenedFilename = fullfile('sharpened', [originalTitle '_sharpened.jpg']);
imwrite(im2uint8(mat2gray(sharpened_image)), sharpenedFilename); % Save sharpened image

wienerFilename = fullfile('wienered', [originalTitle '_wiener_filtered.jpg']);
imwrite(im2uint8(mat2gray(wiener_filtered_image)), wienerFilename); % Save Wiener filtered image

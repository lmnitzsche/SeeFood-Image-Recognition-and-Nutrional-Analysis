% Logan Nitzsche, Tom O'Connell, Sumitra Shrestha, Caleb Sutton,
% RGB
% See Food
% Digital Image Processing - SIUE

% Input image
img = imread('cricket.png');

% Check if the image is RGB (3D array)
if size(img, 3) == 3
    % Convert to grayscale if needed
    img = rgb2gray(img);
end

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
% TODO:
wiener_filtered_image = wiener_filter(img);

% TODO:
% ** Gather image metrics here for experiments **

% Display original, scaled, sharpened, wienered images 
figure;
subplot(1, 4, 1), imshow(img, []), title('Original Image (Grayscale)');
subplot(1, 4, 2), imshow(scaled_image, []), title(['Scaled Image (Factor: ' num2str(scaling_factor) ')']);
subplot(1, 4, 3), imshow(sharpened_image, []), title('Sharpened Image');
subplot(1, 4, 4), imshow(wiener_filtered_image, []), title('Wiener Fitler Image');


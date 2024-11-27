% Load an RGB image
image_rgb = imread('cropfries10.jpg');

% Ensure the image is in RGB format
if size(image_rgb, 3) ~= 3
    error('Input image must be an RGB image.');
end

% Convert the RGB image to indexed image (colormap)
[indexed_image, colormap] = rgb2ind(image_rgb, 256); % Use 256 colors for colormap

% Convert the RGB image to grayscale
image_gray = rgb2gray(image_rgb);

% Simulate Gaussian blur
h = fspecial('gaussian', [5 5], 2); % 7x7 kernel, sigma = 2
blurred_image = imfilter(image_gray, h, 'symmetric');

% Add Gaussian noise (optional, to simulate real conditions)
%noisy_blurred_image = imnoise(blurred_image, 'gaussian', 0, 0.01); % Mean=0, Variance=0.01

% Apply Wiener filter to deblur the image
% Estimate the PSF (Point Spread Function) as the Gaussian kernel
estimated_psf = h; % Gaussian kernel
nsr = 0.01; % Estimated noise-to-signal ratio (adjust as needed)

deblurred_image = deconvwnr(blurred_image, estimated_psf, nsr);

% Reconstruct the RGB image using the colormap
% Normalize the deblurred grayscale image to match the colormap index range
normalized_image = im2uint8(mat2gray(deblurred_image));
reconstructed_image = ind2rgb(normalized_image, colormap);

% Display the results
figure;

subplot(2, 3, 1);
imshow(image_rgb);
title('Original RGB Image');

subplot(2, 3, 2);
imshow(image_gray);
title('Grayscale Image');

subplot(2, 3, 3);
imshow(blurred_image);
title('Blurred Image');

subplot(2, 3, 4);
imshow(deblurred_image, []);
title('Deblurred Image (Wiener Filter)');

subplot(2, 3, 5);
imshow(reconstructed_image);
title('Reconstructed RGB Image');

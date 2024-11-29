function [deblurred_image] = wiener_filter(image_rgb, sigma)

% Ensure the image is in RGB format
if size(image_rgb, 3) ~= 3
    error('Input image must be an RGB image.');
end

if sigma == 1
    h = fspecial('gaussian', [11 11], 1);
end

if sigma == 3
    h = fspecial('gaussian', [11 11], 3);
end

if sigma == 5
    h = fspecial('gaussian', [11 11], 5);
end

estimated_psf = h;
nsr = 0.02;

deblurred_image = deconvwnr(image_rgb, estimated_psf, nsr);
end
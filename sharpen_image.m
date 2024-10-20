function [sharpenedimage] = sharpen_image(img,laplacian_kernal)
    % img: image matrix with non-negative intensity values (uint16)
    % laplacian_kernal: 3x3 matrix used to sharpen image via edge detection
    % returns sharpenedimage: sharpened image matrix 
    %                         with non-negative values (uint16)
    
    % Convolve the image with the Laplacian kernel
    laplacian_filtered = conv2(double(img), laplacian_kernal, 'same');

    % Add result to the original image to sharpen
    sharpenedimage = double(img) + laplacian_filtered;

    % Clip values for uint16 range, convert to uint16
    sharpenedimage = uint16(max(0, min(65535, sharpenedimage)));
end

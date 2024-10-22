function [sharpenedimage] = sharpen_image(img, laplacian_kernal)
    % img: image matrix with non-negative intensity values (uint16)
    % laplacian_kernal: 3x3 matrix used to sharpen image via edge detection
    % returns sharpenedimage: sharpened image matrix with non-negative values (uint16)

    % Get the size of the input image
    [rows, cols] = size(img);
    sharpenedimage = zeros(rows, cols, 'double');
    
    % Zero-padding around the borders of the image
    padded_img = zeros(rows + 2, cols + 2, 'double');
    padded_img(2:rows+1, 2:cols+1) = double(img);
    
    for i = 1:rows
        for j = 1:cols
            % Focus into 3x3 area around the current pixel
            focus_area = padded_img(i:i+2, j:j+2);
            % Convolve with the Laplacian kernel
            result = sum(sum(focus_area .* laplacian_kernal));
            % Add the result to the original pixel value
            sharpened_value = double(img(i, j)) + result;
            % Clip value to uint16 range, add to output image
            sharpenedimage(i, j) = max(0, min(65535, sharpened_value));
        end
    end
    % Convert to uint16
    sharpenedimage = uint16(sharpenedimage);
end
function [sharpenedimage] = sharpen_image(img, laplacian_kernal)
    % img: image matrix with non-negative intensity values (uint16)
    % laplacian_kernal: 3x3 matrix used to sharpen image via edge detection
    % returns sharpenedimage: sharpened image matrix with non-negative values (uint16)
 
    sharpenedimage = zeros(size(img), 'double');
    
    % Process each color channel separately
    for channel = 1:3
        img_channel = img(:, :, channel);
        [rows, cols] = size(img_channel);
        sharpened_channel = zeros(rows, cols, 'double');
        
        % Zero-padding around the borders of the channel
        padded_channel = zeros(rows + 2, cols + 2, 'double');
        padded_channel(2:rows+1, 2:cols+1) = double(img_channel);
        
        for i = 1:rows
            for j = 1:cols
                % Focus into 3x3 area around the current pixel
                focus_area = padded_channel(i:i+2, j:j+2);
                % Convolve with the Laplacian kernel
                result = sum(sum(focus_area .* laplacian_kernal));
                % Add the result to the original pixel value
                sharpened_value = double(img_channel(i, j)) + result;
                % Clip value to uint16 range, add to output channel
                sharpened_channel(i, j) = max(0, min(65535, sharpened_value));
            end
        end
        
        % Store the sharpened channel in the output image
        sharpenedimage(:, :, channel) = sharpened_channel;
    end
    
    % Convert to uint16
    sharpenedimage = uint16(sharpenedimage);
end

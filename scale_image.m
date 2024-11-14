function [scaledimage] = scale_image(img, scalar)
    % img: image matrix with non-negative intensity values
    % scalar: scalar value, double
    % returns scaledimage: scaled image matrix with non-negative values
    
    % Get current rows and columns from img
    [rows, cols, ~] = size(img);
    
    % Scale by scalar value and round to whole number
    rows_scaled = round(rows * scalar);
    cols_scaled = round(cols * scalar);

    % Initialize output image matrix with new dimensions
    scaledimage = zeros(rows_scaled, cols_scaled, 3, 'uint16');

    % Create the nearest neighbor indices for the scaled image
    row_indices = floor((0:rows_scaled - 1) / scalar) + 1;
    col_indices = floor((0:cols_scaled - 1) / scalar) + 1;

    % Use periodicity to wrap around pixels
    row_indices = mod(row_indices - 1, rows) + 1;
    col_indices = mod(col_indices - 1, cols) + 1;

    % Scale each color channel separately to scaled image
    for channel = 1:3
        scaledimage(:, :, channel) = img(row_indices', col_indices, channel);
    end
end

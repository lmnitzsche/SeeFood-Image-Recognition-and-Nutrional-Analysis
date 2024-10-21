function [wiener_filtered_image] = wiener_filter(img)
    % img: image matrix with non-negative intensity values (uint16)

    wiener_filtered_image = wiener2(img, [5 5]);  % Apply 5x5 Wiener filter

end
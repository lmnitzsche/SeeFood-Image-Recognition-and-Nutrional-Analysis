function mse = calculateMSE(original, enhanced) 
    % Get squared differences 
    difference = double(original) - double(enhanced);
    squaredDifference = difference .^ 2;

    % Calculate the MSE
    mse = mean(squaredDifference(:));
end
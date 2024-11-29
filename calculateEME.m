function eme = calculateEME(original, enhanced)
    % original: original image
    % enhanced: enhanced image 
    %  blockSize: Size of the sub-blocks 

    % Initialize variables
    original = double(original);
    enhanced = double(enhanced);
    [rows, cols] = size(original);
    eme = 0;
    numBlocks = 0;
    blockSize = 8;

    % Divide images into blocks
    for i = 1:blockSize:rows - blockSize + 1
        for j = 1:blockSize:cols - blockSize + 1
            % Extract blocks
            blockOriginal = original(i:i+blockSize-1, j:j+blockSize-1);
            blockEnhanced = enhanced(i:i+blockSize-1, j:j+blockSize-1);

            % Calculate contrast enhancement ratio
            minOrig = min(blockOriginal(:)) + 1e-9;
            maxOrig = max(blockOriginal(:));
            minEnh = min(blockEnhanced(:)) + 1e-9;
            maxEnh = max(blockEnhanced(:));

            % Check valid ranges for logs
            if maxOrig > 0 && maxEnh > 0
                eme = eme + log10(maxEnh / minEnh) - log10(maxOrig / minOrig);
                numBlocks = numBlocks + 1;
            end
        end
    end

    % Normalize by the number of blocks
    eme = eme / numBlocks;
end


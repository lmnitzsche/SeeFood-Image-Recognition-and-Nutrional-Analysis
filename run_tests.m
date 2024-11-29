% run tests

% Define directories
baseDir = fullfile(pwd, 'tomset'); 
outputDir = fullfile(pwd, 'output');

% Create output directory if it doesn't exist
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% Get all subfolders in the "tomset" directory
topLevelFolders = dir(baseDir);
topLevelFolders = topLevelFolders([topLevelFolders.isdir] & ~startsWith({topLevelFolders.name}, '.')); % Ignore hidden/system folders

% Iterate through each top-level folder
for i = 1:length(topLevelFolders)
    subFolder1 = fullfile(baseDir, topLevelFolders(i).name);
    
    % Get all subfolders in the current top-level folder
    secondLevelFolders = dir(subFolder1);
    secondLevelFolders = secondLevelFolders([secondLevelFolders.isdir] & ~startsWith({secondLevelFolders.name}, '.')); % Ignore hidden/system folders
    
    for j = 1:length(secondLevelFolders)
        subFolder2 = fullfile(subFolder1, secondLevelFolders(j).name);
        
        % Get all files in the current second-level folder
        imageFiles = dir(fullfile(subFolder2, '*.jpg'));
        
        % Extract the name of the second-level folder
        secondFolderName = secondLevelFolders(j).name;
        
        for k = 1:length(imageFiles)
            % Full path of the image
            imagePath = fullfile(subFolder2, imageFiles(k).name);
            
            % Create new file name based on the structure
            newName = sprintf('%s-%s-%s', ...
                lower(topLevelFolders(i).name), ...
                lower(secondLevelFolders(j).name), ...
                imageFiles(k).name);
            
            % Full path of the new image in the "output" folder
            outputPath = fullfile(outputDir, newName);
            [~, baseName, ext] = fileparts(newName);
            
            % Save the image
            img = imread(imagePath);
            fileID = fopen(fullfile(outputDir, 'results.txt'), 'a');
            % Check the name of the second-level folder
            if strcmpi(secondFolderName, 'Original')
                % Scale
                scaled_image = scale_image(img,10);
                eme = round(calculateEME(img,scaled_image),4);
                fprintf(fileID, 'IMG (SCALED): %s | EME: %.4f\n', baseName, eme);
                % Sharpen
                sharpened_image = sharpen_image(img, [0 -1 0; -1 4 -1; 0 -1 0]);
                eme = round(calculateEME(img,sharpened_image),4);
                fprintf(fileID, 'IMG (SHARPENED): %s | EME: %.4f\n', baseName, eme);
            else
                % The folder is not "Original", attempt to extract "SigmaX"
                sigmaMatch = regexp(secondFolderName, 'Sigma(\d+)$', 'tokens');
                if ~isempty(sigmaMatch)
                    sigma = str2double(sigmaMatch{1}{1});
                    wiener_img = wiener_filter(img,sigma);
                    eme = round(calculateEME(img, wiener_img),4);
                    mse = round(calculateMSE(img, wiener_img),4);
                    fprintf(fileID, 'IMG (WIENER-SIG-%d): %s | EME: %.4f | MSE: %.4f\n', sigma, baseName, eme, mse);
                else
                    sigma = NaN;
                    fprintf('No sigma value found for folder: %s\n', subFolder2);
                end
            end
            fclose(fileID);
        end
    end
end


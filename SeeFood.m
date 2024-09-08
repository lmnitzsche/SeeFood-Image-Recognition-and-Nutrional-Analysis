% math works 978 food images
% greek salad, hamburger, hot dog, pizza, sashimi, and sushi

fprintf("Downloading Example Food Image data set (77 MB)... ")
filename = matlab.internal.examples.downloadSupportFile('nnet', ...
    'data/ExampleFoodImageDataset.zip');
fprintf("Done.\n")

filepath = fileparts(filename);
dataFolder = fullfile(filepath,'ExampleFoodImageDataset');
unzip(filename,dataFolder);



% edaman api

% Define the API endpoint and your credentials
apiKey = '3062a41aac248cbc4eb37ca3c2869c6f'; % Application Key
apiID = 'b705ab51'; % Application ID
baseUrl = 'https://api.edamam.com/api/nutrition-data';

% Define the ingredient to analyze
ingredient = 'apple'; % Example ingredient

% Define the nutrition-type parameter (change to 'logging' if needed)
nutritionType = 'cooking'; 

% Construct the URL with query parameters
queryParams = sprintf('?app_id=%s&app_key=%s&ingr=%s&nutrition-type=%s', apiID, apiKey, urlencode(ingredient), nutritionType);
fullUrl = [baseUrl, queryParams];

% Send the request to the API
options = weboptions('ContentType', 'json', 'RequestMethod', 'get');
try
    response = webread(fullUrl, options);

    % Display and check the available fields in the response
    disp('Nutritional Information:');
    if isfield(response, 'totalNutrients')
        nutrients = response.totalNutrients;
        % Check and display each nutrient if it exists
        if isfield(nutrients, 'ENERC_KCAL')
            disp(['Calories: ', num2str(nutrients.ENERC_KCAL.quantity), ' kcal']);
        end
        if isfield(nutrients, 'PROCNT')
            disp(['Protein: ', num2str(nutrients.PROCNT.quantity), ' g']);
        end
        if isfield(nutrients, 'FASAT')
            disp(['Saturated Fat: ', num2str(nutrients.FASAT.quantity), ' g']);
        end
        if isfield(nutrients, 'CHOCDF')
            disp(['Carbs: ', num2str(nutrients.CHOCDF.quantity), ' g']);
        end
    else
        disp('Nutritional information not available.');
    end
catch ME
    disp(['Error: ', ME.message]);
end


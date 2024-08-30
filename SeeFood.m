% Define the API endpoint and your credentials
apiKey = ''; % Client Secret
apiID = ''; % Client ID
baseUrl = 'https://api.edamam.com/api/food-database/v2/parser';

% Define the food item to search
foodItem = 'apple'; % Example query, replace with actual food item

% Define the nutrition-type parameter
nutritionType = 'cooking';

% Construct the URL with query parameters
queryParams = sprintf('?app_id=%s&app_key=%s&ingr=%s&nutrition-type=%s', apiID, apiKey, urlencode(foodItem), nutritionType);
fullUrl = [baseUrl, queryParams];

% Send the request to the API
options = weboptions('ContentType', 'json', 'RequestMethod', 'get');
try
    response = webread(fullUrl, options);

    % Parse and display the results
    if isfield(response, 'hints')
        for i = 1:length(response.hints)
            food = response.hints(i).food;
            disp(['Food Item: ', food.label]);
            disp(['Nutritional Information:']);
            if isfield(food, 'nutrients')
                disp(['Calories: ', num2str(food.nutrients.ENERC_KCAL), ' kcal']);
                disp(['Protein: ', num2str(food.nutrients.PROCNT), ' g']);
                disp(['Saturated Fat: ', num2str(food.nutrients.FASAT), ' g']);
                disp(['Carbs: ', num2str(food.nutrients.CHOCDF), ' g']);
            else
                disp('Nutritional information not available.');
            end
            disp('---');
        end
    else
        disp('No food items found.');
    end
catch ME
    disp(['Error: ', ME.message]);
end

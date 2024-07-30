% If you want to manually label data in perClassMira then you can click on export labeled data and use the following script to convert it into csv files to train a new model. Use with training.py
% I do not know if you can run this outside of matlab since some of these functions are specific to matlab so just copy this into a script inside of matlab and run it. But make sure to change the path of the file you are saving it to

% Define the path to the .mat file
mat_file_path = 'C:\Users\Headwall\Desktop\BeetleClassifier\labeledData\labeledData.mat';

% Load the .mat file
data = load(mat_file_path);

% Check if the necessary fields exist
if isfield(data, 'data') && ...
   isfield(data.data, 'data') && ...
   isfield(data.data.data, 'prop') && ...
   isfield(data.data.data.prop, 'data') && ...
   isfield(data.data.data.prop, 'class') && ...
   isfield(data.data.data.prop.class, 'data') && ...
   isfield(data.data.data.prop.class.data, 'code')

    % Extract features and labels
    features = data.data.data.prop.data;
    labels = data.data.data.prop.class.data.code;

    % Convert labels from int32 to double if necessary
    labels = double(labels);

    % Save features to CSV file
    csvwrite('C:\Users\Headwall\Desktop\BeetleClassifier\features.csv', features);

    % Save labels to CSV file
    csvwrite('C:\Users\Headwall\Desktop\BeetleClassifier\labels.csv', labels);

    fprintf('Features and labels saved to CSV files.\n');
else
    error('Required fields are missing in the data structure.');
end

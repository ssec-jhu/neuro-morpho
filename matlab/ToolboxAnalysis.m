% Specify the folder containing the MATLAB files
folderPath = '/Users/vkluzner/Work/Projects/NeuralMorphology/Matlab/QNMorph_V1.2/private';

% Get a list of all .m files in the folder
mFiles = dir(fullfile(folderPath, '*.m'));

% Loop through each file and check dependencies
for i = 1:length(mFiles)
    filePath = fullfile(folderPath, mFiles(i).name);
    fprintf(filePath,':\n');

    % Get the required files and products
    [~, productList] = matlab.codetools.requiredFilesAndProducts(filePath);
    
    % Display the file name and the required toolboxes
    fprintf('File: %s\n', mFiles(i).name);
    if ~isempty(productList)
        disp({productList.Name});
    end
    fprintf('\n');
end
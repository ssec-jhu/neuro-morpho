function T = import_time_from_Metadata(filename, startRow, endRow)
    %% Initialize variables.
    delimiter = ',';
    if nargin<=2
        startRow = 1;
        endRow = inf;
    end
    
    %% Format for each line of text:
    %   column1: text (%s)
    %	column2: text (%s)
    %   column3: text (%s)
    % For more information, see the TEXTSCAN documentation.
    formatSpec = '%s%s%s%[^\n\r]';
    
    %% Open the text file.
    fileID = fopen(filename,'r');
    
    %% Read columns of data according to the format.
    % This call is based on the structure of the file used to generate this
    % code. If an error occurs for a different file, try regenerating the code
    % from the Import Tool.
    textscan(fileID, '%[^\n\r]', startRow(1)-1, 'WhiteSpace', '', 'ReturnOnError', false);
    dataArray = textscan(fileID, formatSpec, endRow(1)-startRow(1)+1, 'Delimiter', delimiter, 'TextType', 'string', 'ReturnOnError', false, 'EndOfLine', '\r\n');
    for block=2:length(startRow)
        frewind(fileID);
        textscan(fileID, '%[^\n\r]', startRow(block)-1, 'WhiteSpace', '', 'ReturnOnError', false);
        dataArrayBlock = textscan(fileID, formatSpec, endRow(block)-startRow(block)+1, 'Delimiter', delimiter, 'TextType', 'string', 'ReturnOnError', false, 'EndOfLine', '\r\n');
        for col=1:length(dataArray)
            dataArray{col} = [dataArray{col};dataArrayBlock{col}];
        end
    end
    
    %% Close the text file.
    fclose(fileID);  
    %% Create output variable
Metadata = [dataArray{1:end-1}];
newStr=Metadata(:,1);
newStr1 = extractAfter(newStr,'timestamp #');
newStr2=extractAfter(newStr1,'=');
T = rmmissing(newStr2);
T= str2double(T);
end
    
    

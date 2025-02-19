clear
clc
close all

% %%
% % Set Python environment
% pyenv('Version', '/Library/anaconda3/envs/skelneton/bin/python');
% % Add the directory containing the Python script to the Python path
% if count(py.sys.path, '') == 0
%     insert(py.sys.path, int32(0), '');
% end
% % Call the Python function to load the .pkl file
% file_path = '../../../git/skeletonization/dataloader/annotation/dataset4.pkl';
% file = py.open(file_path, 'rb');
% data = py.pickle.load(file);
% file.close();
% if isa(data, 'py.dict')
%     disp('The data is a Python dictionary.');
% else
%     error('The .pkl file does not contain a dictionary.');
% end
% % Extract keys and values
% keys = py.list(data.keys());  % Convert Python dict keys to MATLAB cell array
% values = py.list(data.values());  % Convert Python dict values to MATLAB cell array
% % Display keys and values
% disp('Keys:');
% disp(keys);
% disp('Values:');
% disp(values);
% 
% % Extract the value associated with the 'val' key
% val_data = data{'val'};
% % Convert Python list to MATLAB cell array
% val_string = string(cell(val_data));
% % Display the result
% disp('Values for "val" key:');
% disp(val_string);
% 
% %%
% 
% N_neuorns=12; % length(val_string);
% addpath(genpath('QNMorph_V1.2'))
% params.WindowType='average';
% params.WindowSize=13;
% params.pixelsize=0.25;
% params.Topology=1;
% params.Fine=1;
% params.Soma=[512,512];%%%%in pixel
% params.persislen_threshold=10.0/params.pixelsize;
% p=parpool('local',N_neuorns);
% 
% parfor ii=1:N_neuorns
% filename=strcat(['../../../../OneDrive/NeuralMorphology/Simulations/' ...
%     'Simulations_16bit_Size1024/output/ex6/submission/'], val_string(ii));
% filename = strrep(filename, '.pgm', '.tif');
% Im=imread(filename);
% 
% BW=logical(Im);
% info=imfinfo(filename);
% Skeletonized_Unet(ii,1)=Scan_Video(BW,Im,params,info);
% end
% delete(p);
% save SkeletonAnalysis_Unet Skeletonized_Unet
% clear var Skeletonized_Unet

%%
clear
clc
close all

N_neurons=10;
addpath(genpath('QNMorph_V1.2.1'))
addpath(genpath('ReadWrite_SWC'))
params.WindowType='average';
params.WindowSize=13;
params.pixelsize=0.25;
params.Topology=1;
params.Fine=1;
params.Soma=[512,512];%%%%in pixel
params.persislen_threshold=10.0/params.pixelsize;

Skeleton = struct();
for cycle=1:10
    fprintf('Cycle %d:\n', cycle)
    p=parpool('local',N_neurons);
    parfor ii=1:N_neurons
        cntr = 10 * (cycle - 1) + ii;
        filename=strcat(['../../../../OneDrive/NeuralMorphology/Simulations/' ...
           'Simulations_16bit_Size1024/images/Skeleton-Sample-'], ...
           num2str(cntr),'-time-36.00.pgm');
        fprintf('%s\n', filename);
        Im=imread(filename);
        BW=logical(Im);
        info=imfinfo(filename);
        try
            Skeletonized(ii,1)=Scan_Video(BW,Im,params,info);
        catch ME
            disp(['Error occured while processing the file ', filename]);
            continue;
        end
        % Convert the result to SWC file
        [filepath, swc_filename, ext] = fileparts(filename);
        idx = find(swc_filename == '-', 1);
        swc_filename = strcat(filepath,'/SWC', swc_filename(idx:end), '_fromSkeleton.swc'); 
        fprintf('%s\n', swc_filename);
        Write_SWCFromMAT(Skeletonized(ii,1), swc_filename);
    end
    if (cycle == 1)
        Skeleton = Skeletonized;
    else
        Skeleton = [Skeleton; Skeletonized];
    end
    delete(p);
end
save SkeletonAnalysis Skeleton
clear var Skeleton Skeletonized

%%
clear
clc
close all

N_neurons=10;
addpath(genpath('QNMorph_V1.2.1'))
addpath(genpath('ReadWrite_SWC'))
params.WindowType='average';
params.WindowSize=13;
params.pixelsize=0.25;
params.Topology=1;
params.Fine=1;
params.Soma=[512,512];%%%%in pixel
params.persislen_threshold=10.0/params.pixelsize;

Segment = struct();
for cycle=1:10
    fprintf('Cycle %d:\n', cycle)
    p=parpool('local',N_neurons);
    parfor ii=1:N_neurons
        cntr = 10 * (cycle - 1) + ii;
        filename=strcat(['../../../../OneDrive/NeuralMorphology/Simulations/' ...
           'Simulations_16bit_Size1024/images/Segmented-Sample-'], ...
           num2str(cntr),'-time-36.00.pgm');
        fprintf('%s\n', filename);
        Im=imread(filename);
        BW=logical(Im);
        info=imfinfo(filename);
        Segmented(ii,1)=Scan_Video(BW,Im,params,info);

        % Convert the result to SWC file
        [filepath, swc_filename, ext] = fileparts(filename);
        idx = find(swc_filename == '-', 1);
        swc_filename = strcat(filepath,'/SWC', swc_filename(idx:end), '_fromSegmented.swc'); 
        fprintf('%s\n', swc_filename);
        Write_SWCFromMAT(Segmented(ii,1), swc_filename);
    end
    if (cycle == 1)
        Segment = Segmented;
    else
        Segment = [Segment; Segmented];
    end
    delete(p);
end

save SegmentedAnalysis Segment
clear var Segment Segmented

%%
clear
clc
close all

N_neurons=10;
addpath(genpath('QNMorph_V1.2.1'))
addpath(genpath('ReadWrite_SWC'))
params.WindowType='average';
params.WindowSize=13;
params.pixelsize=0.25;
params.Topology=1;
params.Fine=1;
params.Soma=[512,512];%%%%in pixel
params.persislen_threshold=10.0/params.pixelsize;

for sbr=1:5
    Real = struct();
    for cycle=1:10
        fprintf('SBR %d, Cycle %d:\n', sbr, cycle)
        p=parpool('local',N_neurons);
        parfor ii=1:N_neurons
            cntr = 10 * (cycle - 1) + ii;
            filename=strcat(['../../../../OneDrive/NeuralMorphology/Simulations/' ...
                'Simulations_16bit_Size1024/images/Realistic-SBR-'],num2str(sbr), ...
                '-Sample-',num2str(cntr),'-time-36.00.pgm');
            fprintf('%s\n', filename);
            Im=imread(filename);
            BW=make_binary(Im,params.WindowSize,params.WindowType);
            info=imfinfo(filename);
            Realistic(ii,1)=Scan_Video(BW,Im,params,info);

            % Convert the result to SWC file
            [filepath, swc_filename, ext] = fileparts(filename);
            dash_indices = find(swc_filename == '-');
            idx = dash_indices(3);
            swc_filename = strcat(filepath,'/SWC', swc_filename(idx:end), ...
                '_fromRealistic-SBR-', num2str(sbr), '.swc'); 
            fprintf('%s\n', swc_filename);
            Write_SWCFromMAT(Realistic(ii,1), swc_filename);
        end
        if (cycle == 1)
            Real = Realistic;
        else
            Real = [Real; Realistic];
        end
        delete(p);
    end
    Mat_workspace = strcat('RealisticAnalysis-', num2str(sbr), '.mat');
    save (Mat_workspace, 'Real')
    clear var Real Realistic
end
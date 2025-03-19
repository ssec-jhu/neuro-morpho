clear
clc
close all

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
% 
% % Convert Python dict keys and values to MATLAB cell array
% keys = py.list(data.keys());
% values = py.list(data.values());
% 
% % Extract the values associated with the 'val' and 'train'keys
% val_data = cell(data{'val'});
% train_data = cell(data{'train'});
% 
% % Convert Python list to MATLAB cell array
% data = sort(string([val_data, train_data]));
% 
% % Display the result
% disp('Values for "val" and "train" keys:');
% disp(data(:));

N_neurons=10;
addpath(genpath('QNMorph_V1.2.1'))
addpath(genpath('ReadWrite_SWC'))
params.WindowType='average';
params.WindowSize=13;
params.pixelsize=0.16;
params.Topology=1;
params.Fine=1;
params.Soma=[1667,1667];%%%%in pixel
params.persislen_threshold=10.0/params.pixelsize;
params.Prune = 0;
params.SaveBinary = 0;
params.SaveSWC = 0;
params.SaveWorkspace = 1;

for sbr=1:5
    Skeleton_Unet = struct();
    for cycle=1:10
        fprintf('SBR %d, Cycle %d:\n', sbr, cycle)
        p=parpool('local',N_neurons);
        parfor ii=1:N_neurons
            cntr = N_neurons * (cycle - 1) + ii;
            filename=strcat(['../../../../OneDrive/NeuralMorphology/' ...
                'Simulations_16bit_Size3334/output/ex8/evaluation/'], ...
                num2str(sbr), '-Sample-', num2str(cntr),'-time-100.00_pred_bin.tif');
            fprintf('%s\n', filename);
            Im=imread(filename);
            BW=logical(Im);
            info=imfinfo(filename);
            try
                Skeletonized_Unet(ii,1)=Scan_Video(BW,Im,params,info);
            catch ME
                Skeletonized_Unet(ii,1) = struct()
                disp(['Error occured while processing the file ', filename]);
                continue;
            end
            
            if (params.SaveSWC == 1) % Convert the result to SWC file
                [filepath, swc_filename, ext] = fileparts(filename);
                swc_filename = strrep(swc_filename, '_pred_bin', '_fromUnet-SBR-');
                idx = find(swc_filename == '-', 1);
                swc_filename = strcat(filepath,'/SWC', swc_filename(idx:end), ...
                    swc_filename(1:idx-1), '.swc');
                fprintf('%s\n', swc_filename);
                Write_SWCFromMAT(Skeletonized_Unet(ii,1), swc_filename);
            end
        end
        if (cycle == 1)
            Skeleton_Unet = Skeletonized_Unet;
        else
            Skeleton_Unet = [Skeleton_Unet; Skeletonized_Unet];
        end
        delete(p);
        clear var Skeletonized_Unet
    end

    if (params.SaveWorkspace == 1)
        Mat_workspace = strcat('SkeletonAnalysis_Unet-', num2str(sbr), '.mat');
        save (Mat_workspace, 'Skeleton_Unet')
    end
    clear var Skeleton_Unet
end

%%
clear
clc
close all

N_neurons=10;
addpath(genpath('QNMorph_V1.2.1'))
addpath(genpath('ReadWrite_SWC'))
params.WindowType='average';
params.WindowSize=13;
params.pixelsize=0.16;
params.Topology=1;
params.Fine=1;
params.Soma=[1667,1667];%%%%in pixel
params.persislen_threshold=10.0/params.pixelsize;
params.Prune = 0;
params.SaveBinary = 0;
params.SaveSWC = 0;
params.SaveWorkspace = 1;

for sbr=1:5
    Prediction_Unet = struct();
    for cycle=1:10
        fprintf('SBR %d, Cycle %d:\n', sbr, cycle)
        p=parpool('local',N_neurons);
        parfor ii=1:N_neurons
            cntr = N_neurons * (cycle - 1) + ii;
            filename=strcat(['../../../../OneDrive/NeuralMorphology/' ...
                'Simulations_16bit_Size3334/output/ex8/evaluation/'], ...
                num2str(sbr), '-Sample-', num2str(cntr),'-time-100.00_pred.tif');
            fprintf('%s\n', filename);
            Im=imread(filename);
            BW=make_binary(Im,params.WindowSize,params.WindowType);
            info=imfinfo(filename);
            try
                Predicted_Unet(ii,1)=Scan_Video(BW,Im,params,info);
            catch ME
                Predicted_Unet(ii,1) = struct()
                disp(['Error occured while processing the file ', filename]);
                continue;
            end
            
            if (params.SaveSWC == 1) % Convert the result to SWC file
                [filepath, swc_filename, ext] = fileparts(filename);
                swc_filename = strrep(swc_filename, '_pred', '_fromUnet_pred-SBR-');
                idx = find(swc_filename == '-', 1);
                swc_filename = strcat(filepath,'/SWC', swc_filename(idx:end), ...
                    swc_filename(1:idx-1), '.swc');
                fprintf('%s\n', swc_filename);
                Write_SWCFromMAT(Predicted_Unet(ii,1), swc_filename);
            end
        end
        if (cycle == 1)
            Prediction_Unet = Predicted_Unet;
        else
            Prediction_Unet = [Prediction_Unet; Predicted_Unet];
        end
        delete(p);
        clear var Predicted_Unet
    end

    if (params.SaveWorkspace == 1)
        Mat_workspace = strcat('PredictedAnalysis_Unet-', num2str(sbr), '.mat');
        save (Mat_workspace, 'Prediction_Unet')
    end
    clear var Prediction_Unet
end

%%
clear
clc
close all

N_neurons=10;
addpath(genpath('QNMorph_V1.2.1'))
addpath(genpath('ReadWrite_SWC'))
params.WindowType='average';
params.WindowSize=13;
params.pixelsize=0.16;
params.Topology=1;
params.Fine=1;
params.Soma=[1667,1667];%%%%in pixel
params.persislen_threshold=10.0/params.pixelsize;
params.Prune = 0;
params.SaveBinary = 0;
params.SaveSWC = 0;
params.SaveWorkspace = 1;

Skeleton = struct();
for cycle=1:10
    fprintf('Cycle %d:\n', cycle)
    p=parpool('local',N_neurons);
    parfor ii=1:N_neurons
        cntr = N_neurons * (cycle - 1) + ii;
        filename=strcat(['../../../../OneDrive/NeuralMorphology/' ...
           'Simulations_16bit_Size3334/images/Skeleton-Sample-'], ...
           num2str(cntr),'-time-100.00.pgm');
        fprintf('%s\n', filename);
        Im=imread(filename);
        BW=logical(Im);
        info=imfinfo(filename);
        try
            Skeletonized(ii,1)=Scan_Video(BW,Im,params,info);
        catch ME
            Skeletonized(ii,1) = struct()
            disp(['Error occured while processing the file ', filename]);
            continue;
        end
        
        if (params.SaveSWC == 1) % Convert the result to SWC file
            [filepath, swc_filename, ext] = fileparts(filename);
            idx = find(swc_filename == '-', 1);
            swc_filename = strcat(filepath,'/SWC', swc_filename(idx:end), '_fromSkeleton.swc'); 
            fprintf('%s\n', swc_filename);
            Write_SWCFromMAT(Skeletonized(ii,1), swc_filename);
        end
    end
    if (cycle == 1)
        Skeleton = Skeletonized;
    else
        Skeleton = [Skeleton; Skeletonized];
    end
    delete(p);
    clear var Skeletonized
end

if (params.SaveWorkspace == 1)
    save SkeletonAnalysis Skeleton
end
clear var Skeleton

%%
clear
clc
close all

N_neurons=10;
addpath(genpath('QNMorph_V1.2.1'))
addpath(genpath('ReadWrite_SWC'))
params.WindowType='average';
params.WindowSize=13;
params.pixelsize=0.16;
params.Topology=1;
params.Fine=1;
params.Soma=[1667,1667];%%%%in pixel
params.persislen_threshold=10.0/params.pixelsize;
params.Prune = 0;
params.SaveBinary = 0;
params.SaveSWC = 0;
params.SaveWorkspace = 1;

Segment = struct();
for cycle=1:10
    fprintf('Cycle %d:\n', cycle)
    p=parpool('local',N_neurons);
    parfor ii=1:N_neurons
        cntr = N_neurons * (cycle - 1) + ii;
        filename=strcat(['../../../../OneDrive/NeuralMorphology/' ...
           'Simulations_16bit_Size3334/images/Segmented-Sample-'], ...
           num2str(cntr),'-time-100.00.pgm');
        fprintf('%s\n', filename);
        Im=imread(filename);
        BW=logical(Im);
        info=imfinfo(filename);
        try
            Segmented(ii,1)=Scan_Video(BW,Im,params,info);
        catch ME
            Segmented(ii,1) = struct()
            disp(['Error occured while processing the file ', filename]);
            continue;
        end

        if (params.SaveSWC == 1) % Convert the result to SWC file
            [filepath, swc_filename, ext] = fileparts(filename);
            idx = find(swc_filename == '-', 1);
            swc_filename = strcat(filepath,'/SWC', swc_filename(idx:end), '_fromSegmented.swc'); 
            fprintf('%s\n', swc_filename);
            Write_SWCFromMAT(Segmented(ii,1), swc_filename);
        end
    end
    if (cycle == 1)
        Segment = Segmented;
    else
        Segment = [Segment; Segmented];
    end
    delete(p);
    clear var Segmented
end

if (params.SaveWorkspace == 1)
    save SegmentedAnalysis Segment
end
clear var Segment

%%
clear
clc
close all

N_neurons=10;
addpath(genpath('QNMorph_V1.2.1'))
addpath(genpath('ReadWrite_SWC'))
params.WindowType='average';
params.WindowSize=13;
params.pixelsize=0.16;
params.Topology=1;
params.Fine=1;
params.Soma=[1667,1667];%%%%in pixel
params.persislen_threshold=10.0/params.pixelsize;
params.Prune = 0;
params.SaveBinary = 0;
params.SaveSWC = 0;
params.SaveWorkspace = 1;

for sbr=1:5
    Real = struct();
    for cycle=1:10
        fprintf('SBR %d, Cycle %d:\n', sbr, cycle)
        p=parpool('local',N_neurons);
        parfor ii=1:N_neurons
        %for ii=1:N_neurons
            cntr = N_neurons * (cycle - 1) + ii;
            filename=strcat(['../../../../OneDrive/NeuralMorphology/' ...
                'Simulations_16bit_Size3334/images/Realistic-SBR-'],num2str(sbr), ...
                '-Sample-',num2str(cntr),'-time-100.00.pgm');
            fprintf('%s\n', filename);
            Im=imread(filename);
            BW=make_binary(Im,params.WindowSize,params.WindowType);
            info=imfinfo(filename);
            try
                Realistic(ii,1)=Scan_Video(BW,Im,params,info);
            catch ME
                Realistic(ii,1) = struct();
                disp(['Error occured while processing the file ', filename]);
                continue;
            end

            if (params.SaveSWC == 1) % Convert the result to SWC file
                [filepath, swc_filename, ext] = fileparts(filename);
                dash_indices = find(swc_filename == '-');
                idx = dash_indices(3);
                swc_filename = strcat(filepath,'/SWC', swc_filename(idx:end), ...
                    '_fromRealistic-SBR-', num2str(sbr), '.swc'); 
                fprintf('%s\n', swc_filename);
                Write_SWCFromMAT(Realistic(ii,1), swc_filename);
            end
        end
        if (cycle == 1)
            Real = Realistic;
        else
            Real = [Real; Realistic];
        end
        delete(p);
        clear var Realistic
    end
    if (params.SaveWorkspace == 1)
        Mat_workspace = strcat('RealisticAnalysis-', num2str(sbr), '.mat');
        save (Mat_workspace, 'Real');
    end
    clear var Real
end
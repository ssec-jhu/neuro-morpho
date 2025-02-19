
%%% load MATLAB analysis of 'Skeleton-Sample-1-time-36.00.tif' saved in Skeleton.mat
%%%and corresponding swc file 'SWC-Sample-1-time-36.00.swc'

%% %%%%%%%%%%%%%%%%%%% lets read the swc file first
filename='SWC-Sample-1-time-36.00.swc';
Tree=Read_SimulatedSWCFile(filename);%%%%%%%%%%%%  READ SWC file and save a tree structure
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plot to make sure things are OK
pixelsize=0.25;
I=imread("Skeleton-Sample-1-time-36.00.tif");

%%%%%% Two things to remember:
% 1. root coordinates are shifted in swc file to be (0,0)
% 2. coordinates are in microns
root_position=[512,512];
%%%%%%%
figure 
imshow(I)
hold on
%%%%%%% branch points
plot(root_position(1)+ceil(Tree.AllBranchPointCoordinate(:,1)./pixelsize),root_position(1)+ceil(Tree.AllBranchPointCoordinate(:,2)./pixelsize),'*r','MarkerSize',10)
%%%ends
plot(root_position(1)+ceil(Tree.AllEndPointCoordinate(:,1)./pixelsize),root_position(1)+ceil(Tree.AllEndPointCoordinate(:,2)./pixelsize),'.b','MarkerSize',20)
%% Now read the analyzed matlab file using QNMorph for 'Skeleton-Sample-1-time-36.00.tif'
%%%and then write a swc file this is SLOW
load SkeletonAnalysis-1.mat
%%%%%%%%%%%%%%%%% Write SWC file %%%%%%%%%%%%%%%%%%%%%%
Write_SWCFromMAT(Skeleton,'SWC-Sample-1-time-36.00_out.swc')

Tree=Read_SimulatedSWCFile('SWC-Sample-1-time-36.00_out.swc');
figure 
imshow(I)
hold on
plot(root_position(1)+ceil(Tree.AllBranchPointCoordinate(:,1)./pixelsize),root_position(1)+ceil(Tree.AllBranchPointCoordinate(:,2)./pixelsize),'*r','MarkerSize',10)
plot(root_position(1)+ceil(Tree.AllEndPointCoordinate(:,1)./pixelsize),root_position(1)+ceil(Tree.AllEndPointCoordinate(:,2)./pixelsize),'.b','MarkerSize',20)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Now read the output swc file and compare

filename='SWC-Sample-1-time-36.00_out.swc';
fid = fopen(filename, 'r'); 
    n = 0;
    while ~feof(fid)
        line = fgetl(fid);
        if startsWith(line, '#')
            continue;
        end
        
        data = sscanf(line, '%d %d %f %f %f %f %d');
        if numel(data) == 7
            n = n + 1;
          
            X(n,1) = data(3);
            Y(n,1) = data(4);           
           
        end
    end
 fclose(fid);   
 
 figure
 imshow(Skeleton.Neuron.skel)
 hold on
 plot(Y/0.25,X/0.25,'*r')
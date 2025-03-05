%%%% FIRST load the real data from the time series generated from the simulation
addpath(genpath('notBoxPlot'))

pix=0.25;
N=100;
k=1;
%%% Read data from the simulation time files
for ii=1:N
    filename = strcat(['../../../../OneDrive/NeuralMorphology/Simulations/' ...
        'Simulations_16bit_Tif_Size1024/TimeData-Sample-'],num2str(ii),'.dat');
    fprintf('%s\n', filename);
    data=importdata(filename);
    TotalL(ii,k)=data.data(end,4);
    NBranches(ii,k)=data.data(end,2);
    NTip(ii,k)=data.data(end,3);
end
%%%%%%%%%%%%%%%% Now LOAD the QNMorph analyzed matlab files for skeleton, segmented and Realistic neurons
%%%%%%%%%%%%%%%%
%%%%%%%%%%% skeleton
k=2;
load('SkeletonAnalysis.mat')
for ii=1:N
    TotalL(ii,k)=sum([Skeleton(ii).Branch.Subtree.TotalLength]).*pix;%%%get the total length
    NBranches(ii,k)=sum([Skeleton(ii).Branch.Subtree.NBranches]);%%%get the total number of branches
    NTip(ii,k)=sum([Skeleton(ii).Branch.Subtree.NTippoints]);%%%get the total number tips
end

%%%%%%%%%%% Do the same for Segmented
k=3;
load('SegmentedAnalysis.mat')
for ii=1:N
    TotalL(ii,k)=sum([Segment(ii).Branch.Subtree.TotalLength]).*pix;
    NBranches(ii,k)=sum([Segment(ii).Branch.Subtree.NBranches]);
    NTip(ii,k)=sum([Segment(ii).Branch.Subtree.NTippoints]);
end

%%%%%%%%%%%Do the same for Realistic
for kk=1:5
    k=k+1;
    Mat_workspace = strcat('RealisticAnalysis-', num2str(kk), '.mat');
    load(Mat_workspace)
    for ii=1:N
        TotalL(ii,k)=sum([Real(ii).Branch.Subtree.TotalLength]).*pix;
        NBranches(ii,k)=sum([Real(ii).Branch.Subtree.NBranches]);
        NTip(ii,k)=sum([Real(ii).Branch.Subtree.NTippoints]);
    end
end

%%%%%%%%%%%Do the same for Unet results
for kk=1:5
    k=k+1;
    Mat_workspace = strcat('SkeletonAnalysis_Unet-', num2str(kk), '.mat');
    load(Mat_workspace)
    for ii=1:N
        if ((k == 9 && ii == 28) || (k == 10 && ii == 84))
            TotalL(ii,k)=TotalL(ii-1,k);
            NBranches(ii,k)=NBranches(ii-1,k);
            NTip(ii,k)=NTip(ii-1,k);
        else
            TotalL(ii,k)=sum([Skeleton_Unet(ii).Branch.Subtree.TotalLength]).*pix;
            NBranches(ii,k)=sum([Skeleton_Unet(ii).Branch.Subtree.NBranches]);
            NTip(ii,k)=sum([Skeleton_Unet(ii).Branch.Subtree.NTippoints]);
        end
    end
end

%% Now Plot the comparisons
%%% I have used an external matlab package for plotting called notBoxPlot
%%link to get it: https://www.mathworks.com/matlabcentral/fileexchange/26508-notboxplot
%%%%%%%%%%%%%%%%%% Now plot Total number of branches
x_axis_lim = k+0.5;
figure
subplot(3,1,1)
notBoxPlot(NBranches)
box on
xlim([0.5,x_axis_lim])
ylabel('Number of branches')
xticklabels({'Gr. truth','Skeleton','Segmented','SBR(1)','SBR(2)','SBR(3)','SBR(4)','SBR(5)','Unet(1)','Unet(2)','Unet(3)','Unet(4)','Unet(5)'})
set(gca,'FontName','Arial','FontSize',16,'LineWidth',1)


subplot(3,1,2)
notBoxPlot(NTip)
box on
xlim([0.5,x_axis_lim])
ylabel('Number of Tips')
xticklabels({'Gr. truth','Skeleton','Segmented','SBR(1)','SBR(2)','SBR(3)','SBR(4)','SBR(5)','Unet(1)','Unet(2)','Unet(3)','Unet(4)','Unet(5)'})
set(gca,'FontName','Arial','FontSize',16,'LineWidth',1)

subplot(3,1,3)
notBoxPlot(TotalL)
box on
xlim([0.5,x_axis_lim])
ylabel('Total Length(\mum)')
xticklabels({'Gr. truth','Skeleton','Segmented','SBR(1)','SBR(2)','SBR(3)','SBR(4)','SBR(5)','Unet(1)','Unet(2)','Unet(3)','Unet(4)','Unet(5)'})
set(gca,'FontName','Arial','FontSize',16,'LineWidth',1)

set(gcf, 'Color','w','Units', 'Inches', 'Position', [0, 0, 6, 12], 'PaperUnits', 'Inches', 'PaperSize', [6, 12])
%saveas(gcf,'AbsComparisons.png')
exportgraphics(gcf, 'AbsComparisons.png', 'Resolution', 300, 'ContentType', 'auto');

%% %%%%%%%%%%%%%%%% Percentage errors 
NN=repmat(NBranches(:,1),1,k);
figure
subplot(3,1,1)
notBoxPlot(100*(NBranches-NN)./NN)
box on
ylabel('% error in branches')
xlim([0.5,x_axis_lim])
xticklabels({'Gr. truth','Skeleton','Segmented','SBR(1)','SBR(2)','SBR(3)','SBR(4)','SBR(5)','Unet(1)','Unet(2)','Unet(3)','Unet(4)','Unet(5)'})
set(gca,'FontName','Arial','FontSize',16,'LineWidth',1)

NN=repmat(NTip(:,1),1,k);
subplot(3,1,2)
notBoxPlot(100*(NTip-NN)./NN)
box on
ylabel('% error in Tips')
xlim([0.5,x_axis_lim])
xticklabels({'Gr. truth','Skeleton','Segmented','SBR(1)','SBR(2)','SBR(3)','SBR(4)','SBR(5)','Unet(1)','Unet(2)','Unet(3)','Unet(4)','Unet(5)'})
set(gca,'FontName','Arial','FontSize',16,'LineWidth',1)

NN=repmat(TotalL(:,1),1,k);
subplot(3,1,3)
notBoxPlot(100*(TotalL-NN)./NN)
box on
ylabel('% error in length')
xlim([0.5,x_axis_lim])
xticklabels({'Gr. truth','Skeleton','Segmented','SBR(1)','SBR(2)','SBR(3)','SBR(4)','SBR(5)','Unet(1)','Unet(2)','Unet(3)','Unet(4)','Unet(5)'})
set(gca,'FontName','Arial','FontSize',16,'LineWidth',1)

set(gcf, 'Color','w','Units', 'Inches', 'Position', [0, 0, 6, 12], 'PaperUnits', 'Inches', 'PaperSize', [6, 12])
%saveas(gcf,'PercentageComparisons.png')
exportgraphics(gcf, 'PercentageComparisons.png', 'Resolution', 300, 'ContentType', 'auto');
%%
edges=(0:2:100)';
centers=(edges(1:end-1)+edges(2:end))/2;
for k=1:20
    filename = strcat(['../../../../OneDrive/NeuralMorphology/Simulations/' ...
        'Simulations_16bit_Tif_Size1024/SWC-Sample-'],num2str(k),'-time-36.00.swc');
    data=importdata(filename);
    NB=max(data.data(:,2));
    
    for ii=1:NB
         X=data.data(data.data(:,2)==ii,3:4);
         a=cumsum(sqrt(diff(X(:,1)).^2+diff(X(:,2)).^2));
        bl{k}(ii,1)=a(end);
    end
end











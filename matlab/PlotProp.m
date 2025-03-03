%%%% FIRST load the real data from the time series generated from the simulation
pix=0.25;
N=20;
k=1;
%%% Read data from the simulation time files
for ii=1:N
    data=importdata(strcat('16-bit/TimeData-Sample-',num2str(ii),'.dat'));
    TotalL(ii,k)=data.data(end,4);
    NBranches(ii,k)=data.data(end,2);
    NTip(ii,k)=data.data(end,3);
end
%%%%%%%%%%%%%%%% Now LOAD the QNMorph analyzed matlab files for skeleton, segmented and Realistic neurons
%%%%%%%%%%%%%%%%
%%%%%%%%%%% skeleton
k=2;
for ii=1:N
    TotalL(ii,k)=sum([Skeletonized(ii).Branch.Subtree.TotalLength]).*pix;%%%get the total length
    NBranches(ii,k)=sum([Skeletonized(ii).Branch.Subtree.NBranches]);%%%get the total number of branches
    NTip(ii,k)=sum([Skeletonized(ii).Branch.Subtree.NTippoints]);%%%get the total number tips
end

%%%%%%%%%%% Do the same for Segmented
k=3;
for ii=1:N
    TotalL(ii,k)=sum([Segmented(ii).Branch.Subtree.TotalLength]).*pix;
    NBranches(ii,k)=sum([Segmented(ii).Branch.Subtree.NBranches]);
    NTip(ii,k)=sum([Segmented(ii).Branch.Subtree.NTippoints]);
end

%%%%%%%%%%%Do the same for Realistic
for kk=1:5
    k=k+1;
for ii=1:N
    TotalL(ii,k)=sum([Realistic(kk,ii).Branch.Subtree.TotalLength]).*pix;
    NBranches(ii,k)=sum([Realistic(kk,ii).Branch.Subtree.NBranches]);
    NTip(ii,k)=sum([Realistic(kk,ii).Branch.Subtree.NTippoints]);
end
end
%% Now Plot the comparisons
%%% I have used an external matlab package for plotting called notBoxPlot
%%link to get it: https://www.mathworks.com/matlabcentral/fileexchange/26508-notboxplot
%%%%%%%%%%%%%%%%%% Now plot Total number of branches
figure
subplot(3,1,1)
notBoxPlot(NBranches)
box on
xlim([0.5,8.5])
ylabel('Number of branches')
xticklabels({'Gr. truth','Skeleton','Segmented','SBR(1)','SBR(2)','SBR(3)','SBR(4)','SBR(5)'})
set(gca,'FontName','Arial','FontSize',16,'LineWidth',1)


subplot(3,1,2)
notBoxPlot(NTip)
box on
xlim([0.5,8.5])
ylabel('Number of Tips')
xticklabels({'Gr. truth','Skeleton','Segmented','SBR(1)','SBR(2)','SBR(3)','SBR(4)','SBR(5)'})
set(gca,'FontName','Arial','FontSize',16,'LineWidth',1)

subplot(3,1,3)
notBoxPlot(TotalL)
box on
xlim([0.5,8.5])
ylabel('Total Length(\mum)')
xticklabels({'Gr. truth','Skeleton','Segmented','SBR(1)','SBR(2)','SBR(3)','SBR(4)','SBR(5)'})
set(gca,'FontName','Arial','FontSize',16,'LineWidth',1)

set(gcf, 'Color','w','Units', 'Inches', 'Position', [0, 0, 6, 12], 'PaperUnits', 'Inches', 'PaperSize', [6, 12])
saveas(gcf,'AbsComparisons.png')
%% %%%%%%%%%%%%%%%% Percentage errors 
NN=repmat(NBranches(:,1),1,8);
figure
subplot(3,1,1)
notBoxPlot(100*(NBranches-NN)./NN)
box on
ylabel('% error in branches')
xlim([0.5,8.5])
xticklabels({'Gr. truth','Skeleton','Segmented','SBR(1)','SBR(2)','SBR(3)','SBR(4)','SBR(5)'})
set(gca,'FontName','Arial','FontSize',16,'LineWidth',1)

NN=repmat(NTip(:,1),1,8);
subplot(3,1,2)
notBoxPlot(100*(NTip-NN)./NN)
box on
ylabel('% error in Tips')
xlim([0.5,8.5])
xticklabels({'Gr. truth','Skeleton','Segmented','SBR(1)','SBR(2)','SBR(3)','SBR(4)','SBR(5)'})
set(gca,'FontName','Arial','FontSize',16,'LineWidth',1)

NN=repmat(TotalL(:,1),1,8);
subplot(3,1,3)
notBoxPlot(100*(TotalL-NN)./NN)
box on
ylabel('% error in length')
xlim([0.5,8.5])
xticklabels({'Gr. truth','Skeleton','Segmented','SBR(1)','SBR(2)','SBR(3)','SBR(4)','SBR(5)'})
set(gca,'FontName','Arial','FontSize',16,'LineWidth',1)

set(gcf, 'Color','w','Units', 'Inches', 'Position', [0, 0, 6, 12], 'PaperUnits', 'Inches', 'PaperSize', [6, 12])
saveas(gcf,'PercentageComparisons.png')
%%
edges=(0:2:100)';
centers=(edges(1:end-1)+edges(2:end))/2;
for k=1:20
data=importdata(strcat("16-bit/SWC-Sample-",num2str(k),"-time-36.00.swc"));
NB=max(data.data(:,2));

for ii=1:NB
     X=data.data(data.data(:,2)==ii,3:4);
     a=cumsum(sqrt(diff(X(:,1)).^2+diff(X(:,2)).^2));
    bl{k}(ii,1)=a(end);
end
end











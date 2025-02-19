function [Neuron,Branch]=Calculate_Properties(I,ws,th);
I(I==0)=100;
BW=make_binary(I,ws,th);
skele=bwmorph(BW,'thin','inf');

branchpoints=bwmorph(skele,'branchpoints');
n_brpts=sum(branchpoints(:));
bdil=imdilate(branchpoints,strel('square',3));
bwdil=imbinarize(double(skele)-double(bdil));

cc=bwconncomp(bwdil);
st=regionprops(cc,'PixelList','PixelIdxList');
totl=0.0;
for ii=1:cc.NumObjects
 b1=zeros(cc.ImageSize);
b1(st(ii).PixelIdxList)=1;
b1=b1+branchpoints;
b2=bwmorph(b1,'bridge');
b3=bwmorph(b2,'close');
bf=bwpropfilt(b3,'Area',1,'largest');
bf1=bwmorph(bf,'thin','inf');
imshow(bf1)
Branch.skel{ii}=create_chain(bf1);
Branch.skel_smoothed{ii}=fit_smoothingspline(Branch.skel{ii}(:,1),Branch.skel{ii}(:,2),size(Branch.skel{ii},1),0.1);
Branch.skelfine{ii} =fit_smoothingspline(Branch.skel_smoothed{ii}(:,1),Branch.skel_smoothed{ii}(:,2),50*size(Branch.skel_smoothed{ii},1),0.05);
 xt=Branch.skelfine{ii}(:,1);yt=Branch.skelfine{ii}(:,2);
 dis=sqrt(diff(xt).^2+diff(yt).^2);
 distance=cumsum(dis);
 Branch.Length(ii,1) =distance(end);
 totl=totl+distance(end);
 clear var dis xt distance 
end
Branch.TotalLength=totl;
%%
figure
imshow(imadjust(I))
hold on
for ii=1:cc.NumObjects
plot(skel_smoothed{ii}(:,1),skel_smoothed{ii}(:,2),'LineWidth',3)
end
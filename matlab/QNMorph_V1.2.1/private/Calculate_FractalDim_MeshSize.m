function [FD,BH,Val]=Calculate_FractalDim_MeshSize(BW)
 [idx,idy]=find(BW==1);
skel=BW(min(idx):max(idx),min(idy):max(idy));

Row=size(skel,1);
Col=size(skel,2);
bw_max=ceil(min([Row,Col])/2);

bw=(4:4:bw_max);

for k=1:length(bw)
nf(k)=0;
nb(k)=0;
nbx=floor(Row/bw(k));
nby=floor(Col/bw(k));
nn=0;
for i=1:nbx
    for j=1:nby
        nn=nn+1;
        imin=(i-1)*bw(k)+1;imax=imin+bw(k)-1;
         jmin=(j-1)*bw(k)+1;jmax=jmin+bw(k)-1;
        if(sum(sum(skel(imin:imax,jmin:jmax)))~=0)
            nf(k)=nf(k)+1;
            nb(k)=nb(k)+1;
        end
    end    
end
 nb(k)=nb(k)/nn;
end
%%
maxr=log10(bw_max);
X=[bw(:),nf(:)];

Y=log10(X);
rmin=min(Y(:,1));
rmax=max(Y(:,1));
df=(rmax-rmin)/3;

Y=Y(Y(:,1)>1.5*df,:);
Y=Y(Y(:,1)<3*df,:);
 [xData, yData] = prepareCurveData( Y(:,1), Y(:,2) );
[fitresult, gof] = fit(xData, yData , 'poly1' );
FD=-fitresult.p1;
% figure
% plot(log10(bw(:)),log10(nf(:)),'ok')
% hold on
% plot(Y(:,1),fitresult(Y(:,1)))
% hold off
% xlabel('Box Size (\mum)')
%% Calculate mesh size 
y=medfilt1(nb(:),3);
ft = 'linearinterp';
[fitresult, gof] = fit( bw(:), y, ft );
BH= arrayfun(@(y)fzero(@(x)fitresult(x)-0.5,1),1);

Val.boxsize=bw(:);
Val.BoxCount=nf(:);
Val.HitProb=nb(:);
%  figure
%  plot(bw(:), nb(:),'-ok')
%  xlabel('Box Size (\mum)')

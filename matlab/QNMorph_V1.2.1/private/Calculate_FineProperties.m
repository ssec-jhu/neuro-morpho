function [FD,BH,FDBHVal]=Calculate_FineProperties(Im)
    if ~islogical(Im)
        BW1=make_binary(Im,21);
        BW1=padarray(BW1,[50,50],0,'both');
        BW= bwareafilt(BW1,1,'largest');
        skele=bwmorph(BW,'thin','inf');
    else
        BW=Im;
        BW=padarray(BW,[50,50],0,'both');
        skele=bwmorph(BW,'thin','inf');
    end
    
    
[idy,idx]=find(skele==1);
cmx=mean(idx(:));
cmy=mean(idy(:));
Rgyr=sqrt(12)*measure_RadiusofGyration(skele);
skelmask=zeros(size(skele));
skeltrim=zeros(size(skele));
rx=Rgyr(1)/2;
ry=Rgyr(2)/2;
xmin=max(1,round(cmy-ry));
xmax=min(round(cmy+ry),size(skele,1));
ymin=max(1,round(cmx-rx));
ymax=min(round(cmx+rx),size(skele,2));
skelmask(xmin:xmax,ymin:ymax)=1;
skeltrim=skele.*skelmask;
skeltrim=logical(skeltrim);
skeltrimf=bwareafilt(skeltrim,1,'largest');
[FD,BH,FDBHVal]=Calculate_FractalDim_MeshSize(skeltrimf);

end

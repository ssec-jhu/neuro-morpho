function [im,xr]=Rotate_Image(image,rads,X)
 xc=X(:,2);yc=X(:,1);   

[Rows, Cols] = size(image); 
Corners=[1,1;1,Cols;Rows,Cols;Rows,1];
RCorners(:,1)=Corners(:,1).*cos(rads)-Corners(:,2).*sin(rads);
RCorners(:,2)=Corners(:,1).*sin(rads)+Corners(:,2).*cos(rads);
maxx=max(RCorners(:,1));minx=min(RCorners(:,1));maxy=max(RCorners(:,2));miny=min(RCorners(:,2));
nrow=max(ceil(maxx-minx+1),Rows);
ncol=max(ceil(maxy-miny+1),Cols);
%%
RowPad =ceil(sqrt(Rows^2+Cols^2));
ColPad =RowPad;
imagepad=image;
imagepad = padarray(image,[RowPad,ColPad],0,'both');
%%
%midpoints
dx=ceil((size(imagepad,1)-Rows)/2);
dy=ceil((size(imagepad,2)-Cols)/2);
imagepad(dx:dx+Rows-1,dy:dy+Cols-1) = image;
%midpoints
midy=ceil((size(imagepad,1))/2);
midx=ceil((size(imagepad,2))/2);
yc=yc+dy-1;
xc=xc+dx-1;
%%

imagerot=zeros(size(imagepad)); % midx and midy same for both
imagerot=cast(imagerot,class(image));

for i=1:size(imagerot,1)
    for j=1:size(imagerot,2)
         xx= (i-midx)*cos(rads)+(j-midy)*sin(rads);
         yy=-(i-midx)*sin(rads)+(j-midy)*cos(rads);
         xx=round(xx)+midx;
         yy=round(yy)+midy;
         if (xx>=1 && yy>=1 && xx<=size(imagepad,1) && yy<=size(imagepad,2))
              imagerot(i,j)=imagepad(xx,yy); % k degrees rotated image         
         end
    end
end

xcr=midx+round( (xc-midx)*cos(rads)-(yc-midy)*sin(rads));
ycr=midy+round( (xc-midx)*sin(rads)+(yc-midy)*cos(rads));

 [idx,idy]=find(imagerot~=0);
im=imagerot(min(idx):max(idx),min(idy):max(idy));
im=padarray(im,[50,50],'both');
ycr=ycr-min(idy)+1+50;
xcr=xcr-min(idx)+1+50;
    
xr=[ycr,xcr];
end



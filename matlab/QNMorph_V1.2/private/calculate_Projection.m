function [Proj, xo]=calculate_Projection(x0,y0,x1,y1,x2,y2)
lv1=sqrt((x0-x1)^2+(y0-y1)^2);
lv2=sqrt((x0-x2)^2+(y0-y2)^2);
    n1x=(x1-x0)/lv1;
    n1y=(y1-y0)/lv1;
    n2x=(x2-x0)/lv1;
    n2y=(y2-y0)/lv1;
    nn=n1x*n2x+n1y*n2y;
   sign=nn/abs(nn);
    projx=x0+(x2-x0)*n1x;
    projy=y0+(y2-y0)*n1y;
    Proj= sign*sqrt((x0-projx)^2+(y0- projy)^2);
    xo(1,1)=x0+Proj*n1x;
     xo(1,2)=y0+Proj*n1y;
    
    
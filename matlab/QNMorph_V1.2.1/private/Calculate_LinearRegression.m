%% function to calculate linear regression

function [m,sigmam,b,sigmab,r]=Calculate_LinearRegression(x,y)
x=x(:);
y=y(:);
n=length(x);
meanx=mean(x);
meany=mean(y);
meanx2=mean(x.^2);
meany2=mean(y.^2);
meanxy=mean(x.*y);
sigmax2=meanx2-meanx^2;sigmax=sqrt(sigmax2);
sigmay2=meany2-meany^2;sigmay=sqrt(sigmay2);
sigmaxy=meanxy-meanx*meany;
r=sigmaxy/(sigmax*sigmay);
m=sigmaxy/sigmax2;
sigmam=sqrt(m^2*(1-r^2)/(r^2*(n-2)));
b=meany-meanx*sigmaxy/sigmax2;
sigmab=sqrt(sigmam^2*(sigmax2+meanx^2));
end
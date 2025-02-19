function XS=fit_smoothingspline(xx,yy,N)
    x=xx(:);
    y=yy(:);
%     if length(x)>2
%     dis=cumsum(sqrt(diff(x).^2+diff(y).^2));
%     arc_distance=dis(end);
%     end_distance=sqrt((x(end)-x(1))^2+(y(end)-y(1))^2);
%     w=1.0-end_distance/arc_distance;
%     else
%         w=0.8;
%     end
% 
%     if w<0.2
%         w=0.2;
%     elseif w>0.8
%         w=0.8;
%     end
w=0.05;    
[x,y]=prepareCurveData(x,y);
n=length(x);
t=(1:n)';
ts=linspace(1,n, N)';
fx = fit(t(:), x(:),'smoothingspline','SmoothingParam',w);
fy = fit(t(:), y(:),'smoothingspline','SmoothingParam',w);

%  fx = fit(t(:), x(:),'smoothingspline');
% fy = fit(t(:), y(:),'smoothingspline');

xp=fx(ts);
yp=fy(ts);
% xp(1)=xx(1);
% xp(end)=xx(end);
% yp(1)=yy(1);
% yp(end)=yy(end);
XS = curvspace([xp(:),yp(:)],N);
%XS=[xp(:),yp(:)];
end


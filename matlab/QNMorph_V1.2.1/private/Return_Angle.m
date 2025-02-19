function th=Return_Angle(cx,cy,X)
if length(X)>=2

xq=[X(1,1),X(end,1)];
yq=[X(1,2),X(end,2)];
%yq=fitr(xq);
% plot(xq,yq,'-g')
% plot([cx xq(1)],[cy yq(1)],'-ob')

u=[xq(1)-cx,yq(1)-cy];
v=[xq(2)-xq(1),yq(2)-yq(1)];

th = atan2( det([u;v]) , dot(u,v) );
else
    th=0/0;
end

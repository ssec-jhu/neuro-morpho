function X=Connect_Points()
global   label points brpic;

ends=bwmorph(brpic,'endpoints');
[endy,endx]=find(ends==1);
 % add first point
 
 x=endx(1);y=endy(1);

points=[x y];
label=zeros(size(brpic));
DFSCluster(y,x);
X=points;
end
%% Depth first search to radially conect points from the center (soma) position
function DFSCluster(i,j)
global brpic label points
label(i,j)= 1;

imin=i-1;
if imin<1
    imin=1;
end
imax=i+1;
if imax>size(brpic,1)
    imax=size(brpic,1);
end
jmin=j-1;
if jmin<1
    jmin=1;
end
jmax=j+1;
if jmax>size(brpic,2)
    jmax=size(brpic,2);
end

for m = imin:imax
    for n=jmin:jmax       
        if  brpic(m,n)==1 && label(m,n) == 0 
            points(end+1,1:2)=[n,m];
            DFSCluster(m,n);
        end
    end
end
end


function X=BFS(object)
global  bw visit xy Q;
bw=object.Neuron.skel;

cc=object.Neuron.APLR(1,:);

mindis=10000000;
for i=cc(2)-15:cc(2)+15
    for j=cc(1)-15:cc(1)+15
        if bw(i,j)==1
            dis=sqrt((cc(2)-i)^2+(cc(1)-j)^2);
            if dis<mindis
                mindis=dis;
                cx=i;
                cy=j;
            end
        end
    end
end

visit=zeros(size(bw));
xy=[cx,cy];
Q=[cx,cy];
visit(cx,cy)=1;
BFSCluster();
X=xy;
end
%% Breadth first search to radially connect points from the center (soma) position
function BFSCluster()
global bw visit xy Q;

while ~isempty(Q)
        % pop first element
    i = queue(1,1);
    j = queue(1,2);
    Q(1,:)=[];%%%%dequeue
visit(i,j)=1;
Q=[i,j];

imin=i-1;
if imin<1
    imin=1;
end
imax=i+1;
if imax>size(bw,1)
    imax=size(bw,1);
end
jmin=j-1;
if jmin<1
    jmin=1;
end
jmax=j+1;
if jmax>size(bw,2)
    jmax=size(bw,2);
end


%%%find neighbors 
kk=0;
for m=imin:imax
    for n=jmin:jmax     
        if bw(m,n)~=0 && visit(m,n)==0
            kk=kk+1;
            neigh(kk,:)=[m,n];
        end
    end
end



for ll=1:kk
    if visit(neigh(ll,1),neigh(ll,2))==0
        xy=vertcat(xy,[neigh(ll,1),neigh(ll,2)]);
        visit(neigh(ll,1),neigh(ll,2))=1;
        Q=vertcat(Q,[neigh(ll,1),neigh(ll,2)]);
       BFSCluster(neigh(ll,1),neigh(ll,2));
end
end
end

end

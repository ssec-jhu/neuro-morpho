% under construction
function [Br,TotalLength,Br_level,Subtree_Size]=Calculate_SubtreeStructure(br,skel,cx,cy)
global  bw label xy queue level brpt;
bw=skel;
bw=bwmorph(bw,'skel','inf');
brpt=bwmorph(bw,'branchpoints');


label=zeros(size(bw));
level=zeros(size(bw));
queue=[];
xy=[cy,cx];
label(cy,cx)=1;
queue(end+1,:)=xy;
level(cy,cx)=1;

NonrecurBFSCluster();
X=xy;
[Br,TotalLength,Br_level,Subtree_Size]=Subtree_Sizes(br,xy,level);
end
%% Breadth first search to radially conect points from the center (soma) position
function NonrecurBFSCluster()
global bw label xy queue level brpt;

while ~isempty(queue)
    % pop first element
    i = queue(1,1);
    j = queue(1,2);
    queue = queue(2:end,:);
    
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
    
    bb=zeros(size(bw));
    bb(imin:imax,jmin:jmax)=bw(imin:imax,jmin:jmax);
    brpts=bb.*brpt;
    
    for m = imin:imax
        for n=jmin:jmax           
            if  bw(m,n)==1 && label(m,n) == 0
                xy(end+1,:)=[m,n];
                label(m,n)= 1;
                queue(end+1,:)=[m,n];
                level(m,n)=max(max(level(imin:imax,jmin:jmax)));
                if(brpts(m,n)==1)
                    level(m,n)=level(m,n)+1;   
                     %level(m-1:m+1,n-1:n+1)=level(m,n).*bw(m-1:m+1,n-1:n+1);                   
                end
 
            end
            
        end
    end
    
end
end

function bwfinal=connect_branchpoint(bw,branchpoints)
brtemp=imbinarize(zeros(size(bw)));
%% find ends
ends=bwmorph(bw,'endpoints');
[endx,endy]=find(ends==1);
%% branchpoint near end 1
lx=endx(1)-2;
if lx<1
    lx=1;
end
ux=endx(1)+2;
if ux>size(bw,2)
    ux=size(bw,2);
end
ly=endy(1)-2;
if ly<1
    ly=1;
end
uy=endy(1)+2;
if uy>size(bw,1)
    uy=size(bw,1);
end
[brx,bry]=find(branchpoints(lx:ux,ly:uy));
min=100000000;
if length(brx)>1
    for i=1:length(brx)
    dis1=sqrt((endx(1)-brx(i))^2+(endy(1)-bry(i))^2);
    if dis1<min
        min=dis1;
        tmpx=brx(i); tmpy=bry(i);
    end
    end
else
    tmpx=brx; tmpy=bry;
end
    brtemp(endx(1)-2+tmpx-1,endy(1)-2+tmpy-1)=1;
    clear var brx bry tmpx tmpy
    %% branchpoint near end 2
lx=endx(2)-2;
if lx<1
    lx=1;
end
ux=endx(2)+2;
if ux>size(bw,2)
    ux=size(bw,2);
end
ly=endy(2)-2;
if ly<1
    ly=1;
end
uy=endy(2)+2;
if uy>size(bw,1)
    uy=size(bw,1);
end
[brx,bry]=find(branchpoints(lx:ux,ly:uy));
    min=100000000;
if length(brx)>1
    for i=1:length(brx)
    dis1=sqrt((endx(2)-brx(i))^2+(endy(2)-bry(i))^2);
    if dis1<min
        min=dis1;
        tmpx=brx(i); tmpy=bry(i);
    end
    end
else
    tmpx=brx; tmpy=bry;
end
brtemp(endx(2)-2+tmpx-1,endy(2)-2+tmpy-1)=1;
  %%
    bwfinal=imbinarize(bw+brtemp);
    bwfinal=bwmorph(bwfinal,'bridge');
     bwfinal=bwmorph(bwfinal,'close');
    bwfinal=bwmorph(bwfinal,'clean');
    bwfinal=bwmorph(bwfinal,'thin','inf');
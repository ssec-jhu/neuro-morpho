function skel=Remove_SinglePixelBranches(skel)
B=bwmorph(skel,'branchpoints');
E= bwmorph(skel,'endpoints');  

EN=zeros(size(E));
[idy,idx]=find(E==1);

for ii=1:length(idx)
    bb=zeros(size(skel));
    imin=idy(ii)-1;
    if(imin<1)
        imin=1;
    end
    imax=idy(ii)+1;
    if imax>size(bb,1)
        imax=size(bb,1);
    end

    jmin=idx(ii)-1;
    if(jmin<1)
        jmin=1;
    end
    jmax=idx(ii)+1;
    if jmax>size(bb,2)
        jmax=size(bb,2);
    end

    bb(imin:imax,jmin:jmax)=B(imin:imax,jmin:jmax);
    if sum(bb(:))>=1
        EN(idy(ii),idx(ii))=1;
    end
    
end
%%
skel=(skel-EN);
skel=bwmorph(skel,'thin','inf');
end
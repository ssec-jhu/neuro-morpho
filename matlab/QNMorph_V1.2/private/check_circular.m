function cc=check_circular(bibr)
    global brpic  
    ends=bwmorph(bibr,'endpoints');
    nend=sum(ends(:));
    ends1=bwmorph(brpic,'endpoints');
    nend1=sum(ends1(:));

if nend1==0%%%%if the branch is circular before adding branchpoint
        bw=brpic;
        ind=find(brpic==1);
        p =randsample(ind,1);
        bw(p)=0;
        cc{1}= bwconncomp(bw);    
elseif nend==0 && nend1~=0 %%if the branch is circular after adding branchpoint;
    if sum(brpic(:))>=3
        bw1=brpic-ends1;
    else
        bw1=brpic;
        ends1=zeros(size(brpic));
    end
        ind=find(bw1==1);
        p =randsample(ind,1);
        bw1(p)=0;  
        bw=imbinarize(bw1+ends1);
%         imshow(bw)
        cc1= bwconncomp(bw);              
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% add the branch points
            for jj=1:cc1.NumObjects       
            bw2=zeros(cc1.ImageSize);
            bw2(cc1.PixelIdxList{jj})=1;
            br=add_branchpoints(bw2);
            cc{jj} = bwconncomp(br);
          end        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    elseif nend<=2%%%%%%%%%% normal caese with 1 or 2 ends
        bw2=bibr;
        cc{1} = bwconncomp(bw2);
    elseif nend>2 %%%%%%% Three or more ends
            [xs,ys]=find(ends==1);
            Dist=pdist([ys(:),xs(:)]);
            Z = squareform(Dist);
            C=triu(Z,1);
            [I,J]=find(C==max(C(:)));  %%% in case there are more than 2 ends find the furthest two end points
            endf=zeros(size(ends));
            endf(xs(I),ys(I))=1;
            endf(xs(J),ys(J))=1;
            bw2=bibr-ends+endf;
            cc{1} = bwconncomp(bw2);
end




end

          
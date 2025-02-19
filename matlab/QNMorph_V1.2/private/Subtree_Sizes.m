%% Orient Branches
function [Br,TotalLength,Br_level,Subtree_Size]=Subtree_Sizes(br,xy,level)
k=0;
nn=0;
n_level=max(level(:));
lv=level;
lv(lv==0)=0/0;
min_level=min(lv(:));

clear var lv;
TotalLength=0.0;
for ii=1:length(br.skel)
    aa=fliplr(br.skel{ii});
    AA=find(ismember(xy,aa,'rows'));
    if ~isempty(AA)
        k=k+1;
        xx=xy(AA,1:2);
        dy1=xx(end,2)-xx(1,2);dx1=xx(end,1)-xx(1,1);
        dy2=aa(end,2)-aa(1,2);dx2=aa(end,1)-aa(1,1);
        if dx1*dx2+dy1*dy2<0  %%%%%positions are in opposite direction
            Br(k).xy=[flipud(aa(:,1)),flipud(aa(:,2))];         
            X=fit_smoothingspline(Br(k).xy(:,1),Br(k).xy(:,2),2*length(Br(k).xy));
            Br(k).xyfine=X;
        else%%%%%positions are in same direction
            Br(k).xy=[aa(:,1),aa(:,2)];
            X=fit_smoothingspline(Br(k).xy(:,1),Br(k).xy(:,2),2*length(Br(k).xy));
            Br(k).xyfine=X;
        end
        aa(aa(:,1)==0,:) = [];
        aa(aa(:,2)==0,:) = [];

        for jj=1:length(aa)
            ll(jj,1)=level(aa(jj,1),aa(jj,2));      
        end
        ll(ll==0)=[];
        Br(k).level=mode(ll);
        Level(k,1)=Br(k).level;
        Br(k).id=ii;
        Br(k).type=br.Type(ii);
        Br(k).angle=Return_Angle(xy(1,2),xy(1,1),fliplr(Br(k).xy));
        dis=sqrt(diff(Br(k).xyfine(:,1)).^2+diff(Br(k).xyfine(:,2)).^2);
        distance=cumsum(dis);
        Br(k).Length=distance(end);
        TotalLength=TotalLength+Br(k).Length;
        clear var ll AA xx X
    end    
end

minlv=min(Level);
for ii=1:length(Br)
    Br(ii).level=Br(ii).level-minlv+1;
end

n_level=max(Level)-minlv+1;
%% organize accroding to levels
clear var Br_level
qq=0;
kk=0;

for jj=1:n_level
    kk=0;
    for ii=1:length(Br)
        if Br(ii).level==jj
            kk=kk+1;
            Br_lv{jj}(kk)= Br(ii);
        end
    end
end
ll=0;
for k=1:length(Br_lv)
    if ~isempty(Br_lv{k})
        ll=ll+1;
        Br_level{ll}=Br_lv{k};
        for l=1:length(Br_level{ll})
            Br_level{ll}(l).level=ll;
        end
    else
        % % fprintf('Empty level %d\n',k);
    end
end
kk=0;

n_level=ll;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   make connection map
for ii=1:length(Br_level)
    for jj=1:length(Br_level{ii})
        Br_level{ii}(jj).parent_id=[];
        Br(Br_level{ii}(jj).id).parent_id=[];
        
        if ii==min_level
            Br_level{ii}(jj).parent_id=0;
            for kk=1:length(Br_level{2})
                Br_level{ii}(jj).offspring_id(kk)=Br_level{min_level+1}(kk).id;
                Br(Br_level{ii}(jj).id).offspring_id(kk)=Br_level{ii}(jj).offspring_id(kk);
            end
        else
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% find parent
            if ii==min_level+1
                Br_level{ii}(jj).parent_id=Br_level{min_level}.id;
                Br(Br_level{ii}(jj).id).parent_id=Br_level{ii}(jj).parent_id;
            else
                mindis=1000000.0;
                for kk=1:length(Br_level{ii-1})
                    x1=Br_level{ii}(jj).xy(1,1); x2=Br_level{ii-1}(kk).xy(end,1);
                    y1=Br_level{ii}(jj).xy(1,2); y2=Br_level{ii-1}(kk).xy(end,2);
                    dis1=sqrt((x1-x2)^2+(y1-y2)^2);
                    if dis1<mindis
                        mindis=dis1;
                        minid=kk;
                    end
                end
                
                if mindis<=sqrt(8)+0.1
                    Br_level{ii}(jj).parent_id=Br_level{ii-1}(minid).id;
                    Br(Br_level{ii}(jj).id).parent_id=Br_level{ii}(jj).parent_id;
                end
                %%%%%%%% did not find any parent then reverse the branch dir
                if isempty(Br_level{ii}(jj).parent_id)
                    mindis=1000000.0;
                    for kk=1:length(Br_level{ii-1})

                        x1=Br_level{ii}(jj).xy(end,1); x2=Br_level{ii-1}(kk).xy(end,1);
                        y1=Br_level{ii}(jj).xy(end,2); y2=Br_level{ii-1}(kk).xy(end,2);

                        dis1=sqrt((x1-x2)^2+(y1-y2)^2);
                        if dis1<mindis
                            mindis=dis1;
                            minid=kk;
                        end
                        
                    end
                    if mindis<=sqrt(8)+0.1
                        Br_level{ii}(jj).parent_id=Br_level{ii-1}(minid).id;
                        Br(Br_level{ii}(jj).id).parent_id=Br_level{ii}(jj).parent_id;
                    end
                end
            end          
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% find offspring
            Br_level{ii}(jj).offspring_id=[];
            Br(Br_level{ii}(jj).id).offspring_id=[];
            if ii<length(Br_level)
                pp=0;
                for kk=1:length(Br_level{ii+1})
                    x1=Br_level{ii}(jj).xy(end,1); x2=Br_level{ii+1}(kk).xy(1,1);
                    y1=Br_level{ii}(jj).xy(end,2); y2=Br_level{ii+1}(kk).xy(1,2);
                    dis=sqrt((x1-x2)^2+(y1-y2)^2);
                    if dis<=sqrt(8)+0.1
                        pp=pp+1;
                        Br_level{ii}(jj).offspring_id(pp)=Br_level{ii+1}(kk).id;
                        Br(Br_level{ii}(jj).id).offspring_id(pp)=Br_level{ii}(jj).offspring_id(pp);
                    end
                end
            end
        end
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%% Calculate Subtree sizes %%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
global q Br_New count ids;
Br_New=Br_level;

for ii=1:length(Br_New)
    for jj=1:length(Br_New{ii})
        %%%%%%%%%% current level %%%%%%%%%%%
        q=[Br_New{ii}(jj).id,Br_New{ii}(jj).level];
        count=0;
        ids=[];     
        ConnectBranches();
        Subtree_Size{ii}(jj,1).count=count;
        Subtree_Size{ii}(jj,1).id=ids;
        %fprintf('level->%d n_level->%d, count=%d\n',ii,jj,count);
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%% Claculate sibling_id %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for ii=1:length(Br)
    Br(ii).sibling_id=[];
end
for ii=1:length(Br)
    if ~isempty(Br(ii).offspring_id) %%%% assuming there are only 2 sisters/binary tree
        sis=[];
        nsis=numel(Br(ii).offspring_id);
        sis=Br(ii).offspring_id;
        for kk=1:nsis
        Br(sis(kk)).sibling_id=sis(sis~=sis(kk));
        end
        
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%% Calculate leaf number &   %%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



for ii=1:length(Br)
    % fprintf("ii=%d, type=%d branch size=%d\n",ii,Br(ii).type,length(Br(ii).xy));
    % if isempty(Br(ii).type)
    %     Br(ii).type=2;
    % end
    Br(ii).LeafNo=[]; %%Initialize with empty vectors
    Br(ii).StrahlerNo=[];
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Label all the tips as 1/leaf #1;
Labeled=0;
for ii=1:length(Br)
    if Br(ii).type==1
    Br(ii).LeafNo=1;
    Br(ii).StrahlerNo=1;
    Labeled=Labeled+1;
    end
    % fprintf("ii=%d, type=%d offspring_id=%d\n",ii,Br(ii).type,isempty(Br(ii).offspring_id));
    if isempty(Br(ii).type) || Br(ii).type==2 && isempty(Br(ii).offspring_id)
    Br(ii).LeafNo=1;
    Br(ii).StrahlerNo=1;
    Labeled=Labeled+1;
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Now the higher order leafs
while Labeled<length(Br)
    for ii=1:length(Br)
        if isempty(Br(ii).LeafNo)
            nd=numel(Br(ii).offspring_id);
            d=[];
            d=Br(ii).offspring_id;
            allval=0;        
            for kk=1:nd
            if ~isempty(Br(d(kk)).LeafNo)
            allval=allval+1; 
            end
            end

            if allval==nd
                strahler=[];
                lf=0;
                for kk=1:nd
                    lf=lf+Br(d(kk)).LeafNo;
                    strahler=vertcat(strahler,reshape(Br(d(kk)).StrahlerNo,[],1));
                end
                Br(ii).LeafNo=lf;

                if max(strahler)==min(strahler)
                     Br(ii).StrahlerNo=max(strahler)+1;
                else
                    Br(ii).StrahlerNo=max(strahler);
                end

                Labeled=Labeled+1;
            end

        end
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
%%
function ConnectBranches()
global q Br_New count ids;
while ~isempty(q)
    id=q(1,1);
    level=q(1,2);
    count=count+1;
    ids(count,1:2)=[id,level];
    q = q(2:end,:);
    %%%%%%%%%%%%%% collect next level ids %%%%%%%
    if level<length(Br_New)
        for ii=1:length(Br_New{level+1})           
            if Br_New{level+1}(ii).parent_id==id
                q=vertcat(q,[Br_New{level+1}(ii).id,Br_New{level+1}(ii).level]);
            end
        end
        ConnectBranches();
    end
end
end
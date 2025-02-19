    function [Neuron,Branch]=Calculate_Properties(Im,ImOri,params)
    Neuron.info=params.info;
    % f=waitbar(0,'Started calculation...');
    [Neuron.FilePath,Neuron.FileName]=fileparts(params.info.Filename);
     Neuron.pixelsize=params.pixelsize;
    global brpts brpic
    %%%%%%%%%%%%%%%%%%%%%%%%%%%% Skeletonize the image
    pad=50;
    Im=padarray(Im,[pad,pad],0,'both');
    BW=logical(Im);
    BW = bwareafilt(BW,1,'largest');
    %%%%%%%%%%%% soma location automatic
    if isfield(params,'Soma')
        Soma_position = params.Soma+pad;
        cx=Soma_position(1);
        cy=Soma_position(2);
    else
        soma_im=bwareafilt(imerode(BW,strel('disk',5)),1,'largest');
        soma_im=imgaussfilt(im2double(soma_im),7);
        soma_im=imbinarize(soma_im);
        soma_im=imfill(soma_im,'holes');
        somas = regionprops(soma_im,'centroid');
        Soma_position = ceil(cat(1,somas.Centroid));
        %%%%%%%%%%%% Merge Soma image to the whole
        BW=BW|soma_im;
    end
      
        %%%%%%%%%%%%%  Skeletonize
        skel=bwmorph(BW,'thin','inf');
        skel=Remove_SinglePixelBranches(skel);
        skel=bwskel(skel,'MinBranchLength',5);
        skel=skel.*logical(BW);
    
        % figure
        % imshow(BW)
        % hold on
        % plot(Soma_position(2),Soma_position(1),'*r')
    
   if ~isfield(params,'Soma')
        %%%%%%%%%%%% Refine soma position to nearest skeleton point
        mindis=10000000;
        for ii=Soma_position(2)-25:Soma_position(2)+25
            for jj=Soma_position(1)-25:Soma_position(1)+25
                if skel(ii,jj)==1
                    dis=sqrt((Soma_position(2)-ii)^2+(Soma_position(1)-jj)^2);
                    if dis<mindis
                        mindis=dis;
                        cx=jj;
                        cy=ii;
                    end
                end
            end
        end
        Soma_position(1)=cx;
        Soma_position(2)=cy;
   end
    
    Neuron.Soma_position=Soma_position-pad;
    
    skel_withSoma=skel;
    
     for ii=cy-3:cy+3
         for jj=cx-3:cx+3
            skel(ii,jj)=0;  
         end
     end
    
    %%%%%%%%%%%%% number of subtrees
    CC0 = bwconncomp(skel);
    % fprintf('%d\n',CC0.NumObjects);
    for n=1:CC0.NumObjects
       skele=zeros(CC0.ImageSize);
       skele(CC0.PixelIdxList{n})=1;
       skele=logical(skele);
      %%%% put soma back
       skele=add_soma(cx,cy,skele);
    
       skele=bwmorph(skele,'thin','inf');
       brpts=bwmorph(skele,'branchpoints');
       endpts=bwmorph(skele,'endpoints');
    % figure
    % imshow(skele)
    % hold on
    % plot(cx,cy,'*b')
    %%%%%%%%%%%%%%%% Branch points
    [idxbr,idybr]=find(brpts==1);
    BR=  [idxbr,idybr];
    [idy,idx]=find(skele==1);
    %%%%%%%%%%%% center of mass location automatic
    cmx=mean(idx(:));
    cmy=mean(idy(:));
    %% remove branchpoints and then break image in isolated branches
    reg_pad=4;
    brpts_dil= imdilate(brpts,strel('square',3));
    skelm=imbinarize(skele-brpts_dil);
    CC1 = bwconncomp(skelm);
    
    %brpts=B;
    totl=0;
    kk=0;
    n_tips=0;
    n_branches=0;
    n_persis=0;
    %      figure
    %      imshowpair(brpts,skele)
    %      hold on
    % f = waitbar(0, 'Starting calculation');
    for ii=1:CC1.NumObjects
        %         fprintf('branch number %d\n',ii);
        b1=zeros(CC1.ImageSize);
        b1(CC1.PixelIdxList{ii})=1;
    
        brpic=imbinarize(b1);
        % imshowpair(brpic,skele)
        bibr=add_branchpoints(brpic);
        cc=check_circular(bibr) ;
    
        for jj=1:length(cc)
            bw=zeros(cc{jj}.ImageSize);
            bw(cc{jj}.PixelIdxList{1})=1;
    
            if sum(bw(:))>=3
                n_branches=n_branches+1;
                kk=kk+1;
                %%%%%%%%%%%%%%%%%%%%%%%%%% Classify as tip(1) or internal branch(2)
                tf=endpts.*bw;
                if(sum(tf(:)))==1
                    Br.Type(kk,1)=1;%%%for internal branches
                    n_tips=n_tips+1;
                else
                    Br.Type(kk,1)=2;%%for tips/terminal branches
                end
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                bw_stat=regionprops(bw,'BoundingBox');
    
                x=ceil(bw_stat.BoundingBox(2)); dx=bw_stat.BoundingBox(4);
                y=ceil(bw_stat.BoundingBox(1)); dy=bw_stat.BoundingBox(3);
                xmin=lower_imlimit(x-reg_pad+1);xmax=upper_imlimit(x+dx+reg_pad-1,size(bw,1));
                ymin=lower_imlimit(y-reg_pad+1);ymax=upper_imlimit(y+dy+reg_pad-1,size(bw,2));
                bwc=bw(xmin:xmax,ymin:ymax);
                brpic=bwc;
    
                Br.skel{kk}=Connect_Points();
                Br.skel{kk}(:,1)=Br.skel{kk}(:,1)+y-reg_pad;
                Br.skel{kk}(:,2)=Br.skel{kk}(:,2)+x-reg_pad;
                %%%%%%%%%%%%%%%%%%%%  Check If the branch is connected to
                %%%%%%%%%%%%%%%%%%%  soma type 0
                A=Br.skel{kk}-pad;
                B=Neuron.Soma_position;
                if ismember(B,A,'rows')
                    Br.Type(kk,1)=0;
                end
                clear var A B
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                clear var b1 bw xt yt X
            end
        end
               % waitbar(ii/CC1.NumObjects, f, sprintf('Calculating branch %d of %d', ii,CC1.NumObjects));
    
    end
    %       fprintf('step 1 Nbranches=%d,N tips=%d\n',kk,n_tips);
    %% %%  add close branchpoints  %%%%%%%%%
    [ys,xs]=find(brpts==1);
    Dist=pdist([ys(:),xs(:)]);
    Z = squareform(Dist);
    C=triu(Z,1);
    C(C==0)=10000;
    [J,I]=find(C<=3*sqrt(2));%%%%distance  bwtween deleted two branchpoins=sqrt(2^2+3^2)
    
    for jj=1:length(I)
        n_branches=n_branches+1;
        kk=kk+1;
        Br.Type(kk,1)=2;
        Br.skel{kk}=[xs(I(jj)) ys(I(jj));xs(J(jj)) ys(J(jj))];
    end
    
    %% %%%%%%%%%%%% orient branches and calculate sub tree properties   
    % waitbar(0.5,f,'orientaing branches...');
    [Br,TotalLength,Branch_level,Subtree_Size]=Calculate_SubtreeStructure(Br,skele,Neuron.Soma_position(1)+pad+1,Neuron.Soma_position(2)+pad+1);
    
    %%%%%%%%%%%%%%%%%%fix branchpoints
    for ii=1:length(Br)
        if length(Br(ii).offspring_id)==2
        child1=Br(ii).offspring_id(1);
        child2=Br(ii).offspring_id(2);
        xm=(Br(ii).xy(end,1)+Br(child1).xy(1,1)+Br(child2).xy(1,1))/3.0;
        ym=(Br(ii).xy(end,2)+Br(child1).xy(1,2)+Br(child2).xy(1,2))/3.0;
        Br(ii).xy(end,1)=xm;
        Br(child1).xy(1,1)=xm;
        Br(child2).xy(1,1)=xm;
        Br(ii).xy(end,2)=ym;
        Br(child1).xy(1,2)=ym;
        Br(child2).xy(1,2)=ym;
        xm=(Br(ii).xyfine(end,1)+Br(child1).xyfine(1,1)+Br(child2).xyfine(1,1))/3.0;
        ym=(Br(ii).xyfine(end,2)+Br(child1).xyfine(1,2)+Br(child2).xyfine(1,2))/3.0;
        Br(ii).xyfine(end,1)=xm;
        Br(child1).xyfine(1,1)=xm;
        Br(child2).xyfine(1,1)=xm;
        Br(ii).xyfine(end,2)=ym;
        Br(child1).xyfine(1,2)=ym;
        Br(child2).xyfine(1,2)=ym;
        end
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Measure diameter
    width=12;
    warning('off');
    % figure
    % imshow(imadjust(ImOri))
    % hold on
    %%%update xy coordinates removing the pad
    for ii=1:length(Br)
        sig=[];
        Br(ii).xy(:,1)=Br(ii).xy(:,1)-pad;
        Br(ii).xy(:,2)=Br(ii).xy(:,2)-pad;
        Br(ii).xyfine(:,1)=Br(ii).xyfine(:,1)-pad;
        Br(ii).xyfine(:,2)=Br(ii).xyfine(:,2)-pad;
        Br(ii).Diam=0/0;
        sigma=[];
      % plot(Br(ii).xyfine(:,2),Br(ii).xyfine(:,1),'.-r')
    
        %%%
        if length(Br(ii).xyfine)>=5
            nn=0;
            delta=max(2,ceil(length(Br(ii).xyfine)/5));
        for kk=3:delta:length(Br(ii).xyfine)-2
            nn=nn+1;
        xm=Br(ii).xyfine(kk,1);
        ym=Br(ii).xyfine(kk,2);
        x1=Br(ii).xyfine(kk-1,1);
        x2=Br(ii).xyfine(kk+1,1);
        y1=Br(ii).xyfine(kk-1,2);
        y2=Br(ii).xyfine(kk+1,2);
    
       dis=sqrt((x2-x1)^2+(y2-y1)^2);
       u1=(x2-x1)/dis;u2=(y2-y1)/dis;%unit vec in parallel
       xv(1)=xm-u2*width/2.0;xv(2)=xm+u2*width/2.0;
       yv(1)=ym+u1*width/2.0;yv(2)=ym-u1*width/2.0;
        % plot(yv,xv,'-g','LineWidth',1)
       inten(:)=improfile(ImOri,yv,xv,2*width,'bicubic');
       [sig ~]= Fitgaussian((1:length(inten))',inten(:));
       sigma(nn,1)=sig;
       % fprintf('Br no=%d Diam=%g\n',ii,sig);
       clear var inten;
        end
      
        % intensity=mean(inten,2,'omitnan');
        Br(ii).Diam=mean(sigma,'omitnan');
        % fprintf('Final Br no=%d Diam=%g\n',ii,Br(ii).Diam);
        %Br(ii).gof=gof;
    
        end
    end
    %%%%%%%%%%%%%%%%%%%%% Measure angles
    dd=3;
    for ii=1:length(Br)
        Br(ii).Angle_Parent=0/0;
        Br(ii).Angle_Sibling=0/0;
            %%%
        if length(Br(ii).xyfine)>dd
        %%%% measure angle with parent
        if length(Br(ii).parent_id)==1
        parentid=Br(ii).parent_id;
    
        x11=Br(parentid).xyfine(end-dd,1);y11=Br(parentid).xyfine(end-dd,2);
        x12=Br(parentid).xyfine(end,1);y12=Br(parentid).xyfine(end,2);
        dist=sqrt((x11-x12).^2+(y11-y12)^2);
        x13=x12+5*(x12-x11)/dist;y13=y12+5*(y12-y11)/dist;
        x21=Br(ii).xyfine(1,1);y21=Br(ii).xyfine(1,2);
        x22=Br(ii).xyfine(dd,1);y22=Br(ii).xyfine(dd,2);
        %plot([y12;y13],[x12;x13],'-k','LineWidth',1)
        %plot([y21;y22],[x21;x22],'-b','LineWidth',1)
        Br(ii).Angle_Parent=calculate_angle(y12,x12,y13,x13,y12,x12,y22,x22)*180/pi;
        end
        %%%% measure angle with sibling
        if length(Br(ii).sibling_id)==1
        siblingid=Br(ii).sibling_id;
        x11=Br(siblingid).xyfine(1,1);y11=Br(siblingid).xyfine(1,2);
        x12=Br(siblingid).xyfine(dd,1);y12=Br(siblingid).xyfine(dd,2);
        x21=Br(ii).xyfine(1,1);y21=Br(ii).xyfine(1,2);
        x22=Br(ii).xyfine(dd,1);y22=Br(ii).xyfine(dd,2);
        %plot([y11;y12],[x11;x12],'-c','LineWidth',2)
        %plot([y21;y22],[x21;x22],'-m','LineWidth',2)
        Br(ii).Angle_Sibling=calculate_angle(y11,x11,y12,x12,y21,x21,y22,x22)*180/pi;
        end
        end
    end
    warning('on');
    % waitbar(0.6,f,'orientaing done...');
    %%%%%%%%%%%%%%%%%%%% other Branch properties
    
    Branch.Subtree(n).skel=Br;
    Branch.Subtree(n).TotalLength=TotalLength;
    Branch.Subtree(n).NBranches=n_branches;
    Branch.Subtree(n).BranchpointPositions=[xs,ys];
    Branch.Subtree(n).NBranchpoints=sum(brpts(:));
    Branch.Subtree(n).NTippoints=n_tips;
    %%%%%%%%%%%%%%%%%%%%%%%% Store topology
    if params.Topology==1
     % waitbar(0.65,f,'Starting topological properties...');
    Branch.Subtree(n).Level=Branch_level;
    Branch.Subtree(n).Subtree=Subtree_Size;
    end
     clear var skele Br
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Neuron fine properties
    Neuron.skel=skel(pad+1:size(skel,1)-pad,pad+1:size(skel,2)-pad);
    Neuron.cmx= cmx-pad;
    Neuron.cmy=cmy-pad;
    prop=regionprops(skel,'Area','ConvexHull','ConvexArea','ConvexImage','EquivDiameter');
    Neuron.Area=prop.Area; % calculate area in um^2
    Neuron.HullArea=prop.ConvexArea;
    Neuron.HullDiam=sqrt(4*Neuron.HullArea/pi);
    Neuron.SkelArea=sum(skel(:));
    Neuron.Rgyr=sqrt(12)*measure_RadiusofGyration(skel);
    Neuron.SkeleDisX=sum(skel,2);
    Neuron.SkeleDisY=sum(skel,1);
    Neuron.R_MassX=Calculate_MassRadius(Neuron.SkeleDisX);
    Neuron.R_MassY=Calculate_MassRadius(Neuron.SkeleDisY);

    %%%%%%%%%%%%%%%%% Calculate Fine properties    
    if params.Fine==1    
    [FD,BH,FDBHVal]=Calculate_FractalDim_MeshSize(skel);
    Neuron.FracDimBox=FD;
    Neuron.MeshSize=BH;
    Neuron.FDBHValues=FDBHVal;
    end   
    end
    %% %%%
    function xx=lower_imlimit(x)
    xx=x;
    if x<1
        xx=1;
    end
    end
    
    function xx=upper_imlimit(x,L)
    xx=x;
    if x>L
        xx=L;
    end
    end
    
    
    function [sigma,gof]= Fitgaussian(xx, yy)
    sigma=0/0;
    gof=[];
    x=xx(:);
    
    %x=x-x(1);
    y=yy(:);
    y(isnan(y))=0;
    y=y-min(y(:));
    y=y./max(y(:));
    xm=x(end)/2;
        
    [xData, yData] = prepareCurveData( x, y );
    
    if length(xData)>=5
    low=[0,0,0.5];
    up=[1,x(end),x(end)];
    start=[0.7,xm,3.0];
    
    ft = fittype( 'a1*exp(-(x-b1).^2./(2*c1^2))', 'independent', 'x', 'dependent', 'y' );
    opts = fitoptions('Method','NonlinearLeastSquares','Upper',up,'Lower',low,'Display','off','StartPoint',start);
    [fitresult, gof] = fit(xData,yData, ft, opts );
    sigma=fitresult.c1;
    
    if gof.rsquare<0.75 
        sigma=0/0;
    end
    else
        sigma=0/0;
        gof=[];
    end
    % if FWHM>20
    % figure
    % plot(x,y,'or')
    % hold on
    % plot(x,fitresult(x),'-k')
    % end
    
    
    % opt = optimoptions('lsqcurvefit','Display','off','MaxIter',1000);
    % g = @(A,X) A(1)*exp(-(X-A(2)).^2./(2*A(3)^2));
    % A0=[1,xm,2.0];
    % AL=[0,xm-5,0.1];
    % AU=[1,xm+5,20,];
    % fitresult = lsqcurvefit(g,A0,x,y,AL,AU,opt);
    % sigma=fitresult(3);
    % FWHM=2.355*sigma;
    end
    
    
    function angle=calculate_angle(x11,y11,x12,y12,x21,y21,x22,y22)
     angle=abs(atan2((x12-x11)*(y22-y21) - (y12-y11)*(x22-x21), (x12-x11)*(x22-x21) + (y12-y11)*(y22-y21)));
    %  if angle >2*pi
    %         angle = angle -2.0 * pi;
    %  end
     %  if angle < 0
     %        angle = angle+ pi;
     % end
    end
    
    function im=add_soma(cx,cy,im)
 
    mindis=10000;
    
     for ii=cy-4:cy+4
         for jj=cx-4:cx+4
             if im(ii,jj)==1
                dis=sqrt((ii-cy)^2+(jj-cx)^2);
                if dis<mindis
                    dis=mindis;
                    imin=ii;
                    jmin=jj;
                end
             end
         end
     end
    
     y(1)=cx;
     y(2)=jmin;
     x(1)=cy;
     x(2)=imin;
    [x,y]=prepareCurveData(x,y);
    n=length(x);
    t=(1:n)';
    ts=linspace(1,n,6)';
    fx = polyfit(t(:), x(:),1);
    fy = polyfit(t(:), y(:),1);
    
    xp=round(polyval(fx,ts));
    yp=round(polyval(fy,ts));
    
    im(cy,cx)=1;
    for ii=1:length(xp)
        im(xp(ii),yp(ii))=1;
    end
    % fprintf("%d %d",imin,jmin);
    end
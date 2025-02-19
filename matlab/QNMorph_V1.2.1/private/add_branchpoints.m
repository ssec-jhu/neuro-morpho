function bwf=add_branchpoints(brpic)
    global brpts
    ends=bwmorph(brpic,'endpoints');
    end1=zeros(size(brpts));
    end2=zeros(size(brpts));
    [endx,endy]=find(ends==1);
     
     if length(endx)>1%
    %%%%%%%%%%%%%%%%%%%%%% end 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end1=connect_ends(endx(1),endy(1));
    end2=connect_ends(endx(2),endy(2));
    %%%%%%%%%%%%%%%%%%%  if there is single end point  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     else
      end1=connect_ends(endx,endy);
     end
    bwf=brpic|end1|end2;
end

function Im=nearset_point(endx,endy,I)
          Im=zeros(size(I));
    if sum(I(:))>1%%%more than one branchpoint in the neighbourhood
          [y,x]=find(I==1);
      dismin=10000000;
      for ii=1:length(x)
         dis=sqrt((endx-x(ii))^2+(endy-y(ii))^2);
         if dis<dismin
             dismin=dis;
             ind=ii;
         end
      end
     Im (y(ind),x(ind))=1;  
    else
       Im=I;  
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

%%
function bf=connect_ends(endx,endy)
global brpts
bf=zeros(size(brpts));
endd=zeros(size(brpts));
endd(endx-2:endx+2,endy-2:endy+2)=1;
br1=brpts.*endd;

if(sum(br1(:))>1)
[x,y]=find(br1==1);
      dismin=10000000;
      for ii=1:length(x)
         dis=sqrt((endx-x(ii))^2+(endy-y(ii))^2);
         if dis<dismin
             dismin=dis;
             ind=ii;
         end
      end
      br1=zeros(size(brpts));
     br1(x(ind),y(ind))=1;   
end


if(sum(br1(:))==1)
[brx,bry]=find(br1==1);

x=[endx;brx];
y=[endy;bry];

t=(1:2)';
ts=(1:0.5:2)';
fx = fit(t(:), x(:),'smoothingspline','SmoothingParam',0);
fy = fit(t(:), y(:),'smoothingspline','SmoothingParam',0);

xp=round(fx(ts));
yp=round(fy(ts));
ind=sub2ind(size(bf),xp,yp);
bf(ind)=1;
end
end

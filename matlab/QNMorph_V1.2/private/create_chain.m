function points=create_chain()
  global brpic
bwdum=brpic;
ends=bwmorph(brpic,'endpoints');
[endy,endx]=find(ends==1);
 % add first point
 x=endx(1);y=endy(1);
 points=[x y];
 % plot(x,y,'*r')

% find all conneted middle points
  while bwdum(y,x)==1
      bwdum(y,x)=0;
      [ dy, dx ] = find( bwdum(y-1:y+1,x-1:x+1)==1 );
      if isempty(dx)
          break
      end
      x = x + dx - 2;
      y = y + dy - 2;
      
      [x,y]=closest_point(points(end,1),points(end,2),x,y);
    points(end+1,1:2)=[x y]; 

  %plot(x,y,'*r')
  end

end

 function [x,y]=closest_point(x0,y0,xx,yy)
    if length(xx)>1
        mindis=1000000;
        for ii=1:length(xx)
            dis=sqrt((x0-xx(ii))^2+(y0-yy(ii))^2);
            if dis<mindis
                dis=mindis;
                x=xx(ii);
                y=yy(ii);
            end
        end
            else
                x=xx;
                y=yy;
        end
    end
            
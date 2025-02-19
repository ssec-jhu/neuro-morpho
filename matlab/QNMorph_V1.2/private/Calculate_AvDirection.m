function [S,Cos_Val]=Calculate_AvDirection(X,ds)
    
 L=cumsum(sqrt(diff(X(:,1)).^2+diff(X(:,2)).^2));
 L_tot=L(end);
 N_p=round(L_tot/ds);
 q=curvspace(X,N_p);
 N_seg=N_p-1;
 ds=L_tot/N_seg;
    
    x=q(:,1);
    y=q(:,2);
    
    
    for i=1:N_seg-1
        S(i,1)=ds*i;
        val=0;
        k=0;
        for j=1:N_seg-i
            %%%% initial segment
            x1=x(j);
            x2=x(j+1);
            y1=y(j);
            y2=y(j+1);
            %%%%%end segment
            x3=x(j+i);
            x4=x(j+i+1);
            y3=y(j+i);
            y4=y(j+i+1);
            
            dx1 = x2-x1;
            dy1 = y2-y1;
            dx2 = x4-x3;
            dy2 = y4-y3;
            d = dx1 *dx2 + dy1 *dy2 ;
            l2 = (dx1^2 + dy1^2) * (dx2^2 + dy2^2) ;
            val =val+ (d/sqrt(l2));
            k=k+1;
        end
        Cos_Val(i,1)=val/k;
    end
    
 
  
    
    
    
    
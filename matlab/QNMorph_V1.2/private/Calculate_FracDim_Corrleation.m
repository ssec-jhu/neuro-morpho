function [FD,FD_gof]=Calculate_FracDim_Corrleation(bw,rmin,rmax,n_rpoints,sampling_count)
    xsize=size(bw,1);
    ysize=size(bw,2);
   % rmax=min(xsize,ysize)/3;
    rr=floor(linspace(rmin,rmax,n_rpoints)');
    
    for ii=1:length(rr)
        r=rr(ii);
        
        [idx,idy]=find(bw==1);
        n_samplepoints=min(sampling_count,numel(idx));
        n_chosen=randperm(numel(idx),n_samplepoints);
        
        n=0;
        for j=1:length(n_chosen)
            bw_cir=zeros(size(bw));
            xc= idx(n_chosen(j));
            yc=idy(n_chosen(j));
            for k=xc-r+1:xc+r
                for l=yc-r+1:yc+r
                    if sqrt((idx(n_chosen(j))-k)^2+(idy(n_chosen(j))-l)^2)<r
                        bw_cir(pbc(k,xsize),pbc(l,ysize))=1;
                    end
                end
            end
            bw_temp2=bw.*bw_cir;
            n=n+sum(bw_temp2(:));
        end
        N(ii,1)=n/n_samplepoints;
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Fit
    rlog=log(rr);
    Nlog=log(N);
    [xData, yData] = prepareCurveData( rlog, Nlog );
    
    % Set up fittype and options.
    ft = fittype( 'poly1' );
    % Fit model to data.
    [fitresult, gof] = fit( xData, yData, ft );
    FD=fitresult.p1;
    FD_gof=gof;
end  
    
    
    function x=pbc(x,sizex)
        if x>sizex
            x=x-sizex;
        end
        if x<1
            x=sizex+x;
        end
    end
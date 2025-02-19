function Inensity=Calculate_Intensity(I,params)
    I(I==0)=100;
    BW=make_binary(I,params.ws,params.scale);
    %%%%%%%% Store Intensity values
    for ii=1:params.n_boxx
        in=(ii-1)*params.boxsize+1;
        for jj=1:params.n_boxy
            jn=(jj-1)*params.boxsize+1;
            xx=I(in:in+params.boxsize-1,jn:jn+params.boxsize-1);
            mask=BW(in:in+params.boxsize-1,jn:jn+params.boxsize-1);
            maskarea=bwarea(mask);
            yy=xx.*uint16(mask);
            Inensity.AvgTot(ii,jj)=mean(xx(:))-100.0;
            Inensity.VarTot(ii,jj)=var(double(xx(:)));
            Inensity.StdTot(ii,jj)=std(double(xx(:)));
            if maskarea==0
                Inensity.AvgMask(ii,jj)=0.0;
                Inensity.VarMask(ii,jj)=0.0;
                Inensity.StdMask(ii,jj)=0.0;
            else
              zz=double(yy(yy~=0));
              Inensity.AvgMask(ii,jj)=mean(zz(:))-100;
              Inensity.VarMask(ii,jj)=var(zz(:));
              Inensity.StdMask(ii,jj)=std(zz(:));
            end
            clear var xx yy mask zz
        end
    end
    
end

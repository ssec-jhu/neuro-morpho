function RGyr=measure_RadiusofGyration(skel)
    [idy,idx]=find(skel==1);
    N_points=sum(skel(:));
    cmx=mean(idx(:));
    cmy=mean(idy(:));

  RGyr(1)=sqrt(sum((idx(:)-cmx).^2)./N_points);
  RGyr(2)=sqrt(sum((idy(:)-cmy).^2)./N_points);
end
  
    
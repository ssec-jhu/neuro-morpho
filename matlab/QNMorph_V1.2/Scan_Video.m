function outfile=Scan_Video(BW,ImFile,params,info)
    
    %% run all images in the tiff stack
    for ii=1:numel(info) 
        params.info=info;
        [out.Neuron,out.Branch]=Calculate_Properties(BW,ImFile,params);   
        outfile(ii)=out;       
    end   
            
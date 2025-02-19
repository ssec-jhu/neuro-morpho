function outfile=Run_Scan(files,paths,params)
if params.Parallel==0
    for ii=1:params.NFile
        ImFile=imread(strcat(paths,files{ii}));
        BW=make_binary(ImFile,params.WindowSize,params.WindowType);
        info=imfinfo(strcat(paths,files{ii}));
        outfile(ii)=Scan_Video(BW,params,info);
    end
else
    delete(gcp('nocreate'))
    c = parcluster(params.PoolName); % build the 'local' cluster object
    nw = c.NumWorkers;        % get the number of workers
    if params.NumCore>nw
        params.NumCore=nw;
    end
for ii=1:params.NFile
info(ii)=imfinfo(strcat(paths,files{ii}));
BW{ii}=make_binary(imread(strcat(paths,files{ii})),params.WindowSize,params.WindowType);
end
    c=parpool(params.PoolName,params.NumCore);
    parfor ii=1:params.NFile    
        outfile(ii)=Scan_Video(BW{ii},params,info(ii));
    end
    delete(c);
end
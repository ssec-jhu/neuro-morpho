function Master_Table=export_csvFile(outfile)
k=0;
for i=1:numel(outfile)
    a=outfile(i).Neuron.pixelsize;
    k=k+1;
    Filename{k,1}=outfile(i).Neuron.FileName;
    AP(k,1)=outfile(i).Neuron.Rgyr(1)*a;
    LR(k,1)=outfile(i).Neuron.Rgyr(2)*a;
    Area(k,1)=AP(k,1)*LR(k,1);
    TotalBranches(k,1)= outfile(i).Branch.NBranches;
    TotalBranchpoints(k,1)=outfile(i).Branch.NBranchpoints;
    TotalTips(k,1)=outfile(i).Branch.NTippoints;
    TotalBrLength(k,1)=outfile(i).Branch.TotalLength*a;
    AvgBrLength(k,1)=TotalBrLength(k,1)/TotalBranches(k,1);
    BrLenDensity(k,1)=TotalBrLength(k,1)/Area(k,1);
    BrDensity(k,1)=TotalBranches(k,1)/Area(k,1);
    
   if isfield(outfile(1).Neuron,'FracDimBox')
    FracDim(k,1)=outfile(i).Neuron.FracDimBox;
    MeshSize(k,1)=outfile(i).Neuron.MeshSize*a ;
   else
    FracDim(k,1)=0/0;
    MeshSize(k,1)=0/0;
   end
end

Master_Table = table(Filename,AP,LR,Area,TotalBranches,TotalBranchpoints,TotalTips,TotalBrLength,AvgBrLength,FracDim,MeshSize,BrLenDensity,BrDensity);
end
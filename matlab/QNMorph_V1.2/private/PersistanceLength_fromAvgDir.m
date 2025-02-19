function [Lp,AvgPersiData]=PersistanceLength_fromAvgDir(D)
    [s,d] = cellfun(@size,D);
    maxrow = max([s,d]);
    n_lengths=length(D);
    AV_Cos=zeros(maxrow,n_lengths);
    AV_DS=zeros(maxrow,n_lengths);
    for nn=1:n_lengths
        AV_DS(1:length(D{nn}),nn)=D{nn}(:,1);
        AV_Cos(1:length(D{nn}),nn)=D{nn}(:,2);
    end
    
    AV_DS(AV_DS==0)=0/0;
    AV_Cos( AV_Cos==0)=0/0;
    X=nanmean(AV_DS,2);
    Y=nanmean(AV_Cos,2);
    AvgPersiData=[X(:),Y(:)];
    Ylog=log(Y);
    n_data=round(maxrow/4);
    [xData, yData] = prepareCurveData(X(1:n_data),Ylog(1:n_data));
    
    % Set up fittype and options.
    ft = fittype( 'poly1' );
    opts = fitoptions( 'Method', 'LinearLeastSquares' );
    % Fit model to data.
    [fitresult, ~] = fit( xData, yData, ft, opts );
    Lp=-1/fitresult.p1;
%     figure
%    plot(fitresult,X,Ylog)
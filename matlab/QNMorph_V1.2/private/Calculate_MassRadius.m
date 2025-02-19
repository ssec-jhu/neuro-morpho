 %% Fit: 'untitled fit 1'.
 function R=Calculate_MassRadius(BWx)
    x=1:length(BWx);
    [xData, yData] = prepareCurveData( x(:), BWx(:));
    
    % Set up fittype and options.
    ft = fittype( 'K*(erf((x-x0+a)./(sig*sqrt(2)))-erf((x-x0-a)./(sig*sqrt(2))));', 'independent', 'x', 'dependent', 'y' );
    opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
    opts.Display = 'Off';
    opts.Lower = [0 0.1 0 0];
    opts.StartPoint = [200 length(BWx)/3 100 length(BWx)/2];
    
    % Fit model to data.
    [fitresult, gof] = fit( xData, yData, ft, opts );
    R=2*(fitresult.a+fitresult.sig);
%     figure
%     plot(fitresult,xData, yData)
%     hold on
%     xx=fitresult.x0-fitresult.a-fitresult.sig;
%     plot(xx,fitresult(xx),'*')
%     xx=fitresult.x0+fitresult.a+fitresult.sig;
%     plot(xx,fitresult(xx),'*')
 
    
    

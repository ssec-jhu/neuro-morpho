function BW= make_binary(I,ws,wt)
% if ws<=5
%     ws=21;
% end

    %%%%%%%%%%%%%%%%%%%%%%%%%% fill blank spaces
    I=padarray(I,[ws,ws],0,'both');

    In=im2double(I);
    nonzero_im=I~=0;
    In=medfilt2(In,[3,3]);
    II=In(In~=0);
    Ic=(In==0);
    %pd=fitdist(II(:),'Normal')
    %Ig=normrnd(pd.mu,pd.sigma,size(In));
    Ig=medfilt2(normrnd(mean(II(:)),2*std(II(:)),size(In)),[3,3]);
 
    In=In+Ic.*Ig;
    %%%%%%%%%%%%%%%%%%%%%%% background subtraction and sharpen the image
   % %Ib=imfilter(In,fspecial('average',ws),'replicate')-imfilter(In,fspecial('average',3),'replicate');
   %  Ib=medfilt2(In,[ws,ws])-medfilt2(In,[1,1]);
   %  %Ib=imgaussfilt(In,ws)-imgaussfilt(In,5);
   %  In = imsubtract(In,Ib);%%%%% subtract the background
   %  In = wiener2(In,[5 5]);%%%local noise removal
   %  In = imsharpen(In,'Radius',8,'Amount',1.5);%%%sharpen the edges
    %%%%%%%%%%%%%% apply Gauss filter for smooth edges
    %In=medfilt2(In,[3,3]);
%%%%%%%%%%%%%%%%%%%%%%% background subtraction and sharpen the image

if strcmp(wt,'Gaussian')==0
    Ib=imgaussfilt(In,ws)-imgaussfilt(In,21);
elseif strcmp(wt,'Median')==0
    if mod(ws,2)==0
        ws=ws+1;
    end
    Ib=medfilt2(In,[ws,ws])-medfilt2(In,[3,3]);
else 
    Ib=imfilter(In,fspecial('average',ws),'replicate')-imfilter(In,fspecial('average',1),'replicate');
    %fprintf('Average selected\n');
end
    In = imsubtract(In,Ib);%%%%% subtract the background
    In = wiener2(In,[5 5]);%%%local noise removal
    In = imsharpen(In,'Radius',8,'Amount',1.5);%%%sharpen the edges


    In=imgaussfilt(In,1);
    pd=fitdist(In(:),'Normal');
    th=pd.mu+0.25*pd.sigma;
    %th1 = adaptthresh(In, 0.15);
    BW = imbinarize(In,th);
    BW=bwareaopen(BW,500);
    BW=bwmorph(BW,'fill');
    BW1=logical(BW .* nonzero_im);
    BW=BW1(ws+1:size(BW1,1)-ws,ws+1:size(BW1,2)-ws);
    BW= bwareafilt(BW,1,'largest');
end

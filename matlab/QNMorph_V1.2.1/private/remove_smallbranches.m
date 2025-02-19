%Function to remove small spurious brances that comes from the
%skeltonization
function I=remove_smallbranches(I,threshold)
filter = [1 1 1;
          1 0 1;
          1 1 1];

I_disconnect = I & ~(I & conv2(double(I), filter, 'same')>2);
cc = bwconncomp(I_disconnect);
numPixels = cellfun(@numel,cc.PixelIdxList);
[sorted_px, ind] = sort(numPixels);
%Remove components shorter than threshold
for ii=ind(sorted_px<=threshold)
    cur_comp = cc.PixelIdxList{ii};
    I(cur_comp) = 0; 
    %Before removing component, check whether image is still connected
    full_cc = bwconncomp(I);
    if full_cc.NumObjects>1
        I(cur_comp) = 1; 
    end
end 
end
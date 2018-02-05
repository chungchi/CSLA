function w_new = set_w(img_num, supixels, x)
addnum = 0;

for i = 1:img_num
    for j = 1:max(supixels{i}(:))
        w_new{i,j} = x(j+addnum,:);
    end
    addnum = addnum + max(supixels{i}(:));
    
    
end

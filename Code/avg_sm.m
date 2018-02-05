function s_avg = avg_sm(supixels, img_num, map_num, Mset)
% gernerate saliency maps
for j = 1:img_num
    spnum(j) = max(supixels{j}(:)); % superpixel numbers
    for q = 1:spnum(j)
        h = supixels{j} == q;
        for t = 1:map_num
        s_avg{j,q}(t) = sum(sum(h.*double(Mset{j,t})))/sum(h(:)); % mean saliency for superpixels
        end
    end;
end;
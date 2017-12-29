function [sigma_c,sigma_g] = compute_sigma_new(img_num, supixels, cfeat, gfeat)
ctotal = 0;
gtotal = 0;
num = 0;

for i = 1:img_num
    for j = 1:max(supixels{i}(:))
        for temp = 1:max(supixels{i}(:))
            cdist = Ka2distance_demo(cfeat{i,j},cfeat{i,temp}); % color distance
            gdist = Ka2distance_demo(gfeat{i,j},gfeat{i,temp}); % gabor distance
            ctotal = ctotal + cdist;
            gtotal = gtotal + gdist;
            num = num + 1;
        end;
        
        if i < img_num
            for ii = (i+1):img_num
                for k = 1: max(supixels{ii}(:))
                    cdist = Ka2distance_demo(cfeat{i,j},cfeat{(ii),k});% color distance
                    gdist = Ka2distance_demo(gfeat{i,j},gfeat{(ii),k});% gabor distance
                    ctotal = ctotal + cdist;
                    gtotal = gtotal + gdist;
                    num = num + 1;
                end;
            end;
        end;
    end;
end;

sigma_c = ctotal/num;
sigma_g = gtotal/num;
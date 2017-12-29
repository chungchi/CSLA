function Mset = read_maps(inames, img_num, map_names, map_num, fdir)
Mset = cell(img_num, map_num);

for m=1:img_num
    for n=1:map_num
        % read-in saliency maps    
        Mset{m,n} = im2double(imread([fdir '/' inames{m} map_names{n}]));
    end; 
end;



 
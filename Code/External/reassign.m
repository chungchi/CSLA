clear;

dirname = './test/0815_bf';
maplist = dir([dirname '/*.png']);

for i=1:length(maplist),
    map = imread([dirname '/' maplist(i).name]);
    map = posdeal2(map,0.4,6);%0.3 is used to control the threshold in equation 13
    % Note that, we use 0.4 for image pair saliency detection, and 0.3 for others. 
    map = normalize(map);
    imwrite(map, ['./test/0815/' maplist(i).name] , 'png');
    disp([maplist(i).name 'is done']);
end;
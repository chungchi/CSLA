function [MAE, AllMAE]= CalMAE(SalMapList, MaskList)
NumImgs = length(MaskList);
AllMAE = zeros(1, NumImgs, 'single');
for i = 1:NumImgs
    AllMAE(i) = mean(abs(im2single(vec(SalMapList{i})) - single(vec(MaskList{i}))));
end
MAE = mean(AllMAE);
end
function [AP,AUC,mF,wf] = DrawPRCurve(SalMapList, MaskList)
NumImages = length(SalMapList);

AllPrec = zeros(NumImages, 256, 'single');
AllRecall = zeros(NumImages, 256, 'single');
AllFP = zeros(NumImages, 256, 'single');
ALLFMEASURE = zeros(NumImages,1, 'single');

for k = 1:NumImages
    [AllPrec(k, :), AllRecall(k, :), AllFP(k,:)] = CalPR(normalize(SalMapList{k}), MaskList{k}, true, true);
    f_measure = WFb(im2double(SalMapList{k}),logical(MaskList{k}));
    ALLFMEASURE(k,1) = f_measure;
end
FMeasure = (1 + 0.3) * AllPrec .* AllRecall ./ (0.3 * AllPrec + AllRecall + eps);
mF = mean(FMeasure, 2);
wf = mean(ALLFMEASURE);

MeanRecall = mean(AllRecall, 1);
MeanPrec = mean(AllPrec , 1);
MeanFP = mean(AllFP, 1);
AP = trapz([0,MeanRecall],[MeanPrec(1),MeanPrec]);
AUC = trapz([0 MeanFP],[MeanRecall(end) MeanRecall]);

end
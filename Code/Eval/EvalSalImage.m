function [MAE, AP, AUC, mF, wf] = EvalSalImage(SalMap, MaskList)
MAE = CalMAE(SalMap, MaskList);
[AP, AUC, mFList, wf] = DrawPRCurve(SalMap, MaskList);
mF = mean(mFList);
end

function [MAE, AllMAE]= CalMAE(SalMapList, MaskList)
NumImgs = length(MaskList);
AllMAE = zeros(1, NumImgs, 'single');
for i = 1:NumImgs
    AllMAE(i) = mean(abs(im2single(vec(SalMapList{i})) - single(vec(MaskList{i}))));
end
MAE = mean(AllMAE);
end

function V = vec(V)
V = V(:);
end

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
AUC = trapz([0 MeanFP],[MeanRecall(1) MeanRecall]);

end

function [Q]= WFb(FG,GT)
% WFb Compute the Weighted F-beta measure (as proposed in "How to Evaluate
% Foreground Maps?" [Margolin et. al - CVPR'14])
% Usage:
% Q = FbW(FG,GT)
% Input:
%   FG - Binary/Non binary foreground map with values in the range [0 1]. Type: double.
%   GT - Binary ground truth. Type: logical.
% Output:
%   Q - The Weighted F-beta score

%Check input
if (~isa( FG, 'double' ))
    error('FG should be of type: double');
end
if ((max(FG(:))>1) || min(FG(:))<0)
    error('FG should be in the range of [0 1]');
end
if (~islogical(GT))
    error('GT should be of type: logical');
end

dGT = double(GT); %Use double for computations.


E = abs(FG-dGT);
% [Ef, Et, Er] = deal(abs(FG-GT));

[Dst,IDXT] = bwdist(dGT);
%Pixel dependency
K = fspecial('gaussian',7,5);
Et = E;
Et(~GT)=Et(IDXT(~GT)); %To deal correctly with the edges of the foreground region
EA = imfilter(Et,K);
MIN_E_EA = E;
MIN_E_EA(GT & EA<E) = EA(GT & EA<E);
%Pixel importance
B = ones(size(GT));
B(~GT) = 2-1*exp(log(1-0.5)/5.*Dst(~GT));
Ew = MIN_E_EA.*B;

TPw = sum(dGT(:)) - sum(sum(Ew(GT)));
FPw = sum(sum(Ew(~GT)));

R = 1- mean2(Ew(GT)); %Weighed Recall
P = TPw./(eps+TPw+FPw); %Weighted Precision

Q = (2)*(R*P)./(eps+R+P); %Beta=1;
% Q = (1+Beta^2)*(R*P)./(eps+R+(Beta.*P));
end

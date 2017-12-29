function Result = EvalSalResult(ReDo, opts)
SaveName = [opts.expDir '/EvalEpoch.mat'];
if ReDo || ~exist(SaveName, 'file')
    MaskList = SetupTrainDataset(opts);
    EpochList = dir([opts.expDir '/net-epoch-*.mat']);
    NumEpoch = length(EpochList);
    Result = struct('MAE', cell(1, NumEpoch), 'AP', cell(1, NumEpoch), ...
        'AUC', cell(1, NumEpoch), 'mF', cell(1, NumEpoch)); % follow "Group-wise Deep Co-saliency Detection"
    disp(['Result Evaluation: ' opts.DatasetName ',' opts.ClassName])
    DisplayStr = ['Epoch:' '%d/' num2str(NumEpoch) '\n'];
    for i = 1:NumEpoch
        fprintf(DisplayStr, i);
        SalMap = load([opts.expDir '/net-epoch-' num2str(i) '.mat' ], 'SalMap');
        SalMap = Im2Uint_Resize(SalMap.SalMap, MaskList);
        [MAE, AP, AUC, mF] = EvalSalImage(SalMap, MaskList);
        Result(i).MAE = mean(MAE);
        Result(i).AP = AP;
        Result(i).AUC = AUC;
        Result(i).mF = mean(mF);
    end
    save(SaveName, 'Result');
else
    Result = load(SaveName);
    Result = Result.Result;
end
end


function NewSalMapList = Im2Uint_Resize(SalMapList, MaskList)
NumImgs = length(MaskList);
NewSalMapList = cell(1, NumImgs);
for i = 1:NumImgs
    NewSalMapList{i} = im2uint8(imresize(SalMapList(:,:,:,i), size(MaskList{i})));
end
end

function GTMask = SetupTrainDataset(opts)
if contains(opts.DatasetName, 'iCoseg') || contains(opts.DatasetName, 'Internet') || contains(opts.DatasetName, 'VOCCoseg')
    MaskExt = '.png';
elseif contains(opts.DatasetName, 'MSRC')
    MaskExt = '.bmp';
end
MaskList = dir([opts.ImageDir '/GroundTruth/*' MaskExt]);
NumImages = length(MaskList);
GTMask = cell(1, NumImages);
RemoveIndex = [];
for i = 1:NumImages
%     disp(['Reading Image: ' num2str(i)])
    [~, ImageName,~] = fileparts(MaskList(i).name);
    TempMask = imread([opts.ImageDir '/GroundTruth/' [ImageName MaskExt]]);
    TempMask = logical(TempMask(:,:,1));
    GTMask{i} = TempMask;
    if all(~TempMask(:))
        RemoveIndex = cat(1, RemoveIndex, i);
    end
end
GTMask(RemoveIndex) = [];
end
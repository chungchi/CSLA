% Data From ICASSP'17
load('Temp.mat');
[MAE, AP, AUC, mF] = EvalSalImage(A, B)
load('Temp1')


% Data From MSRC bike
% CNN
load('Test.mat');
[MAE, AP, AUC, mF] = EvalSalImage(SalMap, MaskList)
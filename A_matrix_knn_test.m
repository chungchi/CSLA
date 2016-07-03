clc; close all; clear all;
% Initial settings
addpath(genpath('./External'));
addpath(genpath('./SACS_code'));
fprintf('begin computation...');
imgDir='./images/pair';
imgDir2='./images/pair/%s';
[Iset,listOfImages]=GetImagePair_demo(imgDir,imgDir2);
gtpath = './cosdata/groundtruth'; % groundtruth path
TIP11result = './cosmap11'; % TIP2011 result
TIP14result = './cosmap14'; % TIP2014 result
local_mappath = './submaps/pair/old/localsaliency'; % local saliency map path
global_mappath = './submaps/pair/old/globalsaliency'; % global saliency map path
map_names = {'_Achanta.bmp','_Itti.bmp','_Hou.bmp','_color.bmp','_patch.bmp'};
map_num = length(map_names);
st=clock;
pridx = 0;       

% Parameter settings
for k = 0:1:8
pridx = pridx+1;

for kk = 1:2:210
display('change image pairs');
kk
par.knn = 2^(k); % coefficient for k-nearest neightbor (1st data)
par.clambda = 1; % affinity value, standard deviation denominator for color feature (2nd data & pairwise)
par.glambda = 2; % affinity value, standard deviation denominator for saliency feature (2nd data & pairwise)
par.ccodebook = 100; % how many bins in color histogram (2nd data & pairwise)
par.cclusternum = 200; % how many run times in kmeans (2nd data & pairwise)
par.numsupixel = 200; % superpixel size for regional fusion [use the same number with TIP2014]

% Self-adaptive weight fusion
display('begin weight fusion')

% read-in image names
imgset = Iset(kk:kk+1);  
img_num = length(imgset);
inames = cell(1,img_num);
imglist = listOfImages(kk:kk+1);
for i=1:img_num,
    inames{i} = strrep(imglist(i).name, '.bmp', ''); % replace string with another
end;

% read-in saliency maps
Mset = cell(img_num, map_num);
fnum = strrep(imglist(1).name, 'a.bmp', '');
fdir1 = sprintf('%s/%s',local_mappath,fnum); % local saliency map directory
fdir2 = sprintf('%s/%s',global_mappath,fnum); % global saliency map directory
for m=1:img_num,
    for n=1:map_num,
        if n<=3
        Mset{m,n} = im2double(imread([fdir1 '/' inames{m} map_names{n}]));
        else
        Mset{m,n} = im2double(imread([fdir2 '/' inames{m} map_names{n}]));
        end
    end; 
end;

% adaptive weight regional fusion for superpixels
[w,supixels] = sacs_calWeight(map_names, inames, Mset, imgDir, par.knn, par.numsupixel); 

% generate null saliency maps
saliency = cell(1,img_num);
for j=1:img_num,
    saliency{j} = zeros(size(Mset{j,1}));
end;

% gernerate saliency maps
for j = 1:img_num
    temp_map = zeros(size(Mset{j,1}));
    for q = 1:max(supixels{j}(:))
        h = supixels{j} == q;
        for t = 1:map_num
        temp_map = temp_map + (w{j,q}(t)*h).*double(Mset{j,t});   
        end
    end;
    saliency{j} = temp_map;
end;

raws = saliency;

for j=1:img_num,
    rs = raws{j};
    map = posdeal2(rs,0.4,6); % [TIP14] use 0.4 for image pair saliency detection, and 0.3 for others. 
    map = normalize(map);
    ours{j} = im2double(map);    
    name = ['results/pairs/'  strrep(imglist(j).name, '.bmp' , '')];
    fdir3 = sprintf('./%s_%i.bmp',name,2^k);
    imwrite(ours{j},fdir3,'bmp');
end;

gt = cell(img_num,1);% read-in groundtruth
TIP11 = cell(img_num,1);% read-in TIP2011 result
TIP14 = cell(img_num,1);% read-in TIP2014 result

for i=1:img_num,
        gt = im2double(imread([gtpath '/' inames{i} '.bmp'])); % read-in groundtruth
        tip11 = im2double(imread([TIP11result '/' inames{i} '.bmp'])); % read-in TIP2011 result
        tip14 = im2double(imread([TIP14result '/' inames{i} '.png'])); % read-in TIP2014 result
        
        if i == 1
        [prec11{kk},tpr11{kk},~,thresh] = prec_rec(double(tip11(:))  ,gt(:),'plotROC',0,'plotPR',0,'numThresh',100);% Precision and Recall
        [prec14{kk},tpr14{kk},~,thresh] = prec_rec(double(tip14(:))  ,gt(:),'plotROC',0,'plotPR',0,'numThresh',100);% Precision and Recall
        [precour{kk},tprour{kk},~,thresh]=prec_rec(double(ours{i}(:)),gt(:),'plotROC',0,'plotPR',0,'numThresh',100);% Precision and Recall
        
        if kk == 1, 
        sum11{pridx} = zeros(numel(prec11{1},1)); % precision
        sum14{pridx} = zeros(numel(prec14{1},1)); % precision
        sumour{pridx}= zeros(numel(precour{1},1)); % precision
        tot11{pridx} = zeros(numel(tpr11{1},1)); % recall
        tot14{pridx} = zeros(numel(tpr14{1},1)); % recall
        totour{pridx}= zeros(numel(tprour{1},1)); % recall
        end
        
        s11  = prec11{kk}; sum11{pridx}  = sum11{pridx} + s11'; % sum precision
        s14  = prec14{kk}; sum14{pridx}  = sum14{pridx} + s14'; % sum precision
        sour = precour{kk};sumour{pridx} = sumour{pridx}+ sour';  % sum precision       
        t11  = tpr11{kk};  tot11{pridx}  = tot11{pridx} + t11'; % total recall
        t14  = tpr14{kk};  tot14{pridx}  = tot14{pridx} + t14'; % total recall
        tour = tprour{kk}; totour{pridx} = totour{pridx}+ tour'; % total recall
        
        else
        [prec11{kk+1},tpr11{kk+1},~,thresh] = prec_rec(double(tip11(:))   ,gt(:),'plotROC',0,'plotPR',0,'numThresh',100);% Precision and Recall
        [prec14{kk+1},tpr14{kk+1},~,thresh] = prec_rec(double(tip14(:))   ,gt(:),'plotROC',0,'plotPR',0,'numThresh',100);% Precision and Recall
        [precour{kk+1},tprour{kk+1},~,thresh]= prec_rec(double(ours{i}(:)),gt(:),'plotROC',0,'plotPR',0,'numThresh',100);% Precision and Recall 
        
        s11  = prec11{kk+1}; sum11{pridx}  = sum11{pridx} + s11'; % sum precision
        s14  = prec14{kk+1}; sum14{pridx}  = sum14{pridx} + s14'; % sum precision
        sour = precour{kk+1};sumour{pridx} = sumour{pridx}+ sour'; % sum precision
        t11  = tpr11{kk+1};  tot11{pridx}  = tot11{pridx} + t11'; % total recall
        t14  = tpr14{kk+1};  tot14{pridx}  = tot14{pridx} + t14'; % total recall
        tour = tprour{kk+1}; totour{pridx} = totour{pridx}+ tour'; % total recall
        end
        
end;

end;
end;
fprintf(' took %.2f minutes\n',etime(clock,st)/60);

% Precision and Recall
display('begin precision recall curve')
% precision-recall curve
figure; hold on; xlabel('recall'); ylabel('precision');
plot(fliplr([tot11{pridx}/((kk+1)),0]),fliplr([sum11{pridx}/((kk+1)),1]), 'y'); % yellow
plot(fliplr([tot14{pridx}/((kk+1)),0]),fliplr([sum14{pridx}/((kk+1)),1]), 'y-*'); % yellow
plot(fliplr([totour{1}/(kk+1),0]),fliplr([sumour{1}/(kk+1),1]), 'c'); % cyan
plot(fliplr([totour{2}/(kk+1),0]),fliplr([sumour{2}/(kk+1),1]), 'c-*'); % cyan
plot(fliplr([totour{3}/(kk+1),0]),fliplr([sumour{3}/(kk+1),1]), 'r'); % red
plot(fliplr([totour{4}/(kk+1),0]),fliplr([sumour{4}/(kk+1),1]), 'r-*'); % red
plot(fliplr([totour{5}/(kk+1),0]),fliplr([sumour{5}/(kk+1),1]), 'g'); % green
plot(fliplr([totour{6}/(kk+1),0]),fliplr([sumour{6}/(kk+1),1]), 'g-*'); % black
plot(fliplr([totour{7}/(kk+1),0]),fliplr([sumour{7}/(kk+1),1]), 'b'); % blue
plot(fliplr([totour{8}/(kk+1),0]),fliplr([sumour{8}/(kk+1),1]), 'b-*'); % white
plot(fliplr([totour{9}/(kk+1),0]),fliplr([sumour{9}/(kk+1),1]), 'k'); % black
% plot(fliplr([totour{10}/(kk+1),0]),fliplr([sumour{10}/(kk+1),1]), 'k-*'); % white
% plot(fliplr([totour{11}/(kk+1),0]),fliplr([sumour{11}/(kk+1),1]), 'm'); % black
% plot(fliplr([totour{12}/(kk+1),0]),fliplr([sumour{12}/(kk+1),1]), 'm-*'); % white
% legend('tip11','tip14','pridx1','pridx2','pridx3','pridx4','pridx5','pridx6','pridx7','pridx8','pridx9','pridx10','pridx11','pridx12');
grid on; hold off;
% attention****
% area under the precision-recall curve
Inttip11 = trapz(fliplr([tot11{pridx}/((kk+1)),0]), fliplr([sum11{pridx}/((kk+1)),1]));
Inttip14 = trapz(fliplr([tot14{pridx}/((kk+1)),0]), fliplr([sum14{pridx}/((kk+1)),1]));
Int1 = trapz(fliplr([totour{1}/(kk+1),0]),fliplr([sumour{1}/(kk+1),1])); % cyan
Int2 = trapz(fliplr([totour{2}/(kk+1),0]),fliplr([sumour{2}/(kk+1),1])); % red
Int3 = trapz(fliplr([totour{3}/(kk+1),0]),fliplr([sumour{3}/(kk+1),1])); % green
Int4 = trapz(fliplr([totour{4}/(kk+1),0]),fliplr([sumour{4}/(kk+1),1])); % blue
Int5 = trapz(fliplr([totour{5}/(kk+1),0]),fliplr([sumour{5}/(kk+1),1])); % white
Int6 = trapz(fliplr([totour{6}/(kk+1),0]),fliplr([sumour{6}/(kk+1),1])); % black
Int7 = trapz(fliplr([totour{7}/(kk+1),0]),fliplr([sumour{7}/(kk+1),1])); % white
Int8 = trapz(fliplr([totour{8}/(kk+1),0]),fliplr([sumour{8}/(kk+1),1])); % black
Int9 = trapz(fliplr([totour{9}/(kk+1),0]),fliplr([sumour{9}/(kk+1),1])); % white
% Int10 = trapz(fliplr([totour{10}/(kk+1),0]),fliplr([sumour{10}/(kk+1),1])); % black
% Int11 = trapz(fliplr([totour{11}/(kk+1),0]),fliplr([sumour{11}/(kk+1),1])); % white
% Int12 = trapz(fliplr([totour{12}/(kk+1),0]),fliplr([sumour{12}/(kk+1),1])); % black
% jj = sprintf('%.4i/%.4i/%.4i/%.4i/%.4i/%.4i/%.4i/%.4i/%.4i/%.4i',Int1,Int2,Int3,Int4,Int5,Int6,Int7,Int8,Int9,Int10);
% title((jj))
% AxesH = gcf;   % Not the GCF
% F = getframe(AxesH);
% fdir5 = sprintf('./%s/%s/%s.png','results','emap','auc');
% imwrite(F.cdata,fdir5,'png');


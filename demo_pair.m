clc; close all; clear all;

addpath(genpath('./External')); % add function path
addpath(genpath('./SACS_code')); % add function path

TIP11result = './tip11'; % TIP2011 result path
TIP14result = './tip14'; % TIP2014 result path
ACCV14result = './accv14'; % ACCV14 result path
gtpath = './cosdata/groundtruth'; % groundtruth path
imgDir='./cosdata/images/pair'; % input image path
imgDir2='./cosdata/images/pair/%s'; % input image path
local_mappath = './submaps/pair/old/localsaliency'; % local saliency maps path
global_mappath = './submaps/pair/old/globalsaliency'; % global saliency maps path

map_names = {'_Achanta.bmp','_Itti.bmp','_Hou.bmp','_color.bmp','_patch.bmp'};
map_num = length(map_names); 

%% Parameter settings
st=clock;
fprintf('begin computation...');
[Iset,listOfImages]=GetImagePair_demo(imgDir,imgDir2); % input test images

display('change image pairs');
for kk = 1:2:20
kk
pridx = 0;       
par.knn = 2^(0); % coefficient for k-nearest neightbor (1st data)
par.clambda = 1; % affinity value, standard deviation denominator for color feature (2nd data & pairwise)
par.glambda = 2; % affinity value, standard deviation denominator for saliency feature (2nd data & pairwise)
par.numsupixel = 200; % superpixel size for regional fusion [use the same number with TIP2014]
par.ccodebook = 100; % how many bins in color histogram (2nd data & pairwise)
par.cclusternum = 200; % how many run times in kmeans (2nd data & pairwise)
%% Self-adaptive weight fusion
display('begin accv14 for A');

% read-in image names
imgset = Iset(kk:kk+1);  
img_num = length(imgset);
inames = cell(1,img_num);
imglist = listOfImages(kk:kk+1);
inames{1} = strrep(imglist(1).name,'.bmp',''); % replace string with another
inames{2} = strrep(imglist(2).name,'.bmp',''); % replace string with another

% supixels{1} = SLIC_mex( imgset{1}, 200, 20 ); % SLIC superpixels 
% supixels{2} = SLIC_mex( imgset{2}, 200, 20 ); % SLIC superpixels
supixels{1} = LSC_mex( imgset{1}, 200, 0.075 ); % LSC superpixels
supixels{2} = LSC_mex( imgset{2}, 200, 0.075 ); % LSC superpixels

% read-in saliency maps
Mset = cell(img_num, map_num);
imgnum = strrep(imglist(1).name,'a.bmp','');
fdir1 = sprintf('%s/%s',local_mappath,imgnum); % local maps path
fdir2 = sprintf('%s/%s',global_mappath,imgnum); % global maps path
Mset = read_maps(inames, img_num, map_names, map_num, fdir1, fdir2); 

% gernerate saliency maps
for j = 1:img_num
    spnum(j) = max(supixels{j}(:)); % superpixel numbers
    X = zeros(spnum(j),map_num);
    for q = 1:spnum(j)
        h = supixels{j} == q;
        for t = 1:map_num
        s_avg{j,q}(t) = sum(sum(h.*double(Mset{j,t})))/sum(h(:)); % mean saliency for superpixels
        X(q,t) = s_avg{j,q}(t);
        end
    end;
    lamda = 0.06; % parameter in ACCV14
    [~ ,E] = exact_alm_rpca(X,lamda); % use augmented largrangian mutltiplier RPCA 
    S{j} = double(E); % take the residual as saliency
end;

% saliency_accv14 = save_accv14_result(img_num,Mset,supixels,S,imglist,spnum);

%% Generating color and gabor features 
display('begin generating features');

cvector = []; % color vector
gvector = []; % gabor vector

for i=1:img_num,       
    IRGB = imread([imgDir '/' inames{i},'.bmp']);
    RGB = im2double(IRGB);
    R = RGB(:,:,1);
    G = RGB(:,:,2);
    B = RGB(:,:,3);
    Ycbr = double(rgb2ycbcr(IRGB));
    Y = Ycbr(:,:,1)/255;
    Cb = Ycbr(:,:,2)/255;
    Cr = Ycbr(:,:,3)/255;
    [Cl,Ca,Cb] = rgb2lab(IRGB(:,:,1),IRGB(:,:,2),IRGB(:,:,3));
    Lab = [Cl(:)/100,(Ca(:)+110)/220,(Cb(:)+110)/220];
    rgb = [R(:),G(:),B(:)];
    ybr = [Y(:),Cb(:),Cr(:)];
    cvecc{i} = [rgb,Lab,ybr]; % color feature
    cvector = [cvector;cvecc{i}];
    gvecc{i} = create_feature_space(IRGB); % gabor feature
    gvector = [gvector;gvecc{i}];
end

% K-means clustering for color features
des = cvector;
randid = floor(size(des,1)*rand(1,par.ccodebook))'+1;
randcen = des(randid,:);
[ccenv,ccenl] = do_kmeans(des',par.ccodebook,par.cclusternum, randcen');
ccenl = ccenl+1;

% K-means clustering for gabor features
des = gvector;
randid = floor(size(des,1)*rand(1,par.ccodebook))'+1;
randcen = des(randid,:);
[gcenv,gcenl] = do_kmeans(des',par.ccodebook,par.cclusternum, randcen');
gcenl = gcenl+1;

% generate the feature histogram for superpixels
for i=1:img_num, 
    if (i == 1)
      for j = 1:max(supixels{1}(:))    
          idx = find(supixels{i}(:) == j);
          cfeat{i,j} = hist(ccenl(idx),(1:par.ccodebook))/numel(idx);
          gfeat{i,j} = hist(gcenl(idx),(1:par.ccodebook))/numel(idx);
      end;
    elseif (i == 2)
      for j = 1:max(supixels{2}(:))     
          idx = find(supixels{i}(:) == j);
          cfeat{i,j} = hist(ccenl(idx+numel(supixels{1})),(1:par.ccodebook))/numel(idx);
          gfeat{i,j} = hist(gcenl(idx+numel(supixels{1})),(1:par.ccodebook))/numel(idx);
      end;   
    end       
end
%% Parameter settings
for par_setting1 = 0 % beta1
for par_setting2 = -5:1 % alpha1
for par_setting3 = 0 % alpha2
for par_setting4 = 0 % k nearest neighbors
pridx = pridx + 1;
close all; clc;
display([par_setting1,par_setting2,par_setting3,par_setting4])
par.beta1 = 2^(par_setting1); % coefficient for pairwise term
par.alpha1 = 2^(par_setting2); % coefficient for second data term
par.alpha2 = 2^(par_setting3); % coefficient for third data term 
par.largest_p_value = 2^(par_setting4); % number of inter-image responses (pairwise)

%% First data term
display('begin A matrix');
A = set_A(img_num, map_num, spnum, S);

%% Affinity matrix
display('begin affinity matrix');
% Computer the normalization constant
[sigma_c,sigma_g] = compute_sigma(img_num, supixels, cfeat, gfeat);
% compute the affinity matrix
[M, e1, e2, e] = build_affinity(supixels,img_num,gfeat,cfeat,par,sigma_c,sigma_g);

% Calculating laplacian matrix for pairwise term
display('begin Laplacian matrix');
affinity = M;
clear D
% Compute the degree matrix
for i=1:size(affinity,1)
    D(i,i) = sum(affinity(i,:));
end

L = D - affinity;
NL = D^(-1/2) * L * D^(-1/2);

%% Second data term
display('begin 2nd data term');
B = set_B(e,e1,e2,s_avg,img_num,spnum);

%% Optimization
display('begin optimization');

Nsum = max(supixels{1}(:))+max(supixels{2}(:));
clear x;
cvx_begin quiet
    variable x(Nsum,map_num)
    minimize(par.alpha1*trace(A*x') + par.alpha2*trace(B*x') + par.beta1*sum(quad_form(x(:,1),NL)+quad_form(x(:,2),NL)+quad_form(x(:,3),NL)+quad_form(x(:,4),NL)+quad_form(x(:,5),NL)) + ...
        sum(quad_form(x(:,1),eye(Nsum))+quad_form(x(:,2),eye(Nsum))+quad_form(x(:,3),eye(Nsum))+quad_form(x(:,4),eye(Nsum))+quad_form(x(:,5),eye(Nsum))))             
    subject to 
    ones(1,map_num)*x' == ones(1,Nsum);
    0 <= x(:) <= 1
cvx_end

%% Optimiztion weight
display('comparing pre(post) optimization weight');
w_new = set_w(img_num, supixels, x);

%% Cosalieny maps
display('begin fusion maps generation')

for j=1:img_num,
    saliency{j} = zeros(size(Mset{j,1}));
end;

for j = 1:img_num
    temp_map = zeros(size(Mset{j,1}));
    for q=1:spnum(j)
        h = supixels{j} == q;
        for t = 1:map_num
        temp_map = temp_map + (w_new{j,q}(t)*h).*double(Mset{j,t});   
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
    name = ['test/pairs/'  strrep(imglist(j).name, '.bmp' , '')];
    fdir4 = sprintf('./%s_%i.bmp',name,pridx);
    imwrite(ours{j},fdir4,'bmp');
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
end;
end;
end;
fprintf(' took %.2f minutes\n',etime(clock,st)/60);

%% Precision and Recall
display('begin precision recall curve');
% precision-recall curve
figure; hold on; xlabel('recall'); ylabel('precision');
plot(fliplr([tot11{pridx}/((kk+1)),0]),fliplr([sum11{pridx}/((kk+1)),1]), 'y'); % yellow 
plot(fliplr([tot14{pridx}/((kk+1)),0]),fliplr([sum14{pridx}/((kk+1)),1]), 'y-*'); % yellow
plot(fliplr([totour{1}/(kk+1),0]),fliplr([sumour{1}/(kk+1),1]), 'c'); % cyan(1)
plot(fliplr([totour{2}/(kk+1),0]),fliplr([sumour{2}/(kk+1),1]), 'c-*'); % cyan(2)
plot(fliplr([totour{3}/(kk+1),0]),fliplr([sumour{3}/(kk+1),1]), 'r'); % red(3)
plot(fliplr([totour{4}/(kk+1),0]),fliplr([sumour{4}/(kk+1),1]), 'r-*'); % red(4)
plot(fliplr([totour{5}/(kk+1),0]),fliplr([sumour{5}/(kk+1),1]), 'g'); % green(5)
plot(fliplr([totour{6}/(kk+1),0]),fliplr([sumour{6}/(kk+1),1]), 'g-*'); % black(6)
plot(fliplr([totour{7}/(kk+1),0]),fliplr([sumour{7}/(kk+1),1]), 'b'); % blue(7)
% plot(fliplr([totour{8}/(kk+1),0]),fliplr([sumour{8}/(kk+1),1]), 'b-*'); % white(8)
% plot(fliplr([totour{9}/(kk+1),0]),fliplr([sumour{9}/(kk+1),1]), 'k'); % black(9)
% plot(fliplr([totour{10}/(kk+1),0]),fliplr([sumour{10}/(kk+1),1]), 'k-*');% white(10)
% plot(fliplr([totour{11}/(kk+1),0]),fliplr([sumour{11}/(kk+1),1]), 'm'); % black(11)
% plot(fliplr([totour{12}/(kk+1),0]),fliplr([sumour{12}/(kk+1),1]), 'm-*');% white(12)
% legend('tip11','tip14','pridx1','pridx2','pridx3','pridx4','pridx5','pridx6','pridx7','pridx8','pridx9','pridx10','pridx11','pridx12');
grid on; hold off;
% attention****
% area under the precision-recall curve
Inttip11 = trapz(fliplr([tot11{pridx}/((kk+1)),0]), fliplr([sum11{pridx}/((kk+1)),1]));
Inttip14 = trapz(fliplr([tot14{pridx}/((kk+1)),0]), fliplr([sum14{pridx}/((kk+1)),1]));
Int1 = trapz(fliplr([totour{1}/(kk+1),0]),fliplr([sumour{1}/(kk+1),1])); % cyan(1)
Int2 = trapz(fliplr([totour{2}/(kk+1),0]),fliplr([sumour{2}/(kk+1),1])); % red(2)
Int3 = trapz(fliplr([totour{3}/(kk+1),0]),fliplr([sumour{3}/(kk+1),1])); % green(3)
Int4 = trapz(fliplr([totour{4}/(kk+1),0]),fliplr([sumour{4}/(kk+1),1])); % blue(4)
Int5 = trapz(fliplr([totour{5}/(kk+1),0]),fliplr([sumour{5}/(kk+1),1])); % white(5)
Int6 = trapz(fliplr([totour{6}/(kk+1),0]),fliplr([sumour{6}/(kk+1),1])); % black(6)
Int7 = trapz(fliplr([totour{7}/(kk+1),0]),fliplr([sumour{7}/(kk+1),1])); % white(7)
% Int8 = trapz(fliplr([totour{8}/(kk+1),0]),fliplr([sumour{8}/(kk+1),1])); % black(8)
% Int9 = trapz(fliplr([totour{9}/(kk+1),0]),fliplr([sumour{9}/(kk+1),1])); % white(9)
% Int10 = trapz(fliplr([totour{10}/(kk+1),0]),fliplr([sumour{10}/(kk+1),1])); % black(10)
% Int11 = trapz(fliplr([totour{11}/(kk+1),0]),fliplr([sumour{11}/(kk+1),1])); % white(11)
% Int12 = trapz(fliplr([totour{12}/(kk+1),0]),fliplr([sumour{12}/(kk+1),1])); % black(12)
% jj = sprintf('%.4i/%.4i/%.4i/%.4i/%.4i/%.4i/%.4i/%.4i/%.4i/%.4i',Int1,Int2,Int3,Int4,Int5,Int6,Int7,Int8,Int9,Int10);
% title((jj))
% AxesH = gcf;   % Not the GCF
% F = getframe(AxesH);
% fdir5 = sprintf('./%s/%s/%s.png','results','emap','auc');
% imwrite(F.cdata,fdir5,'png');


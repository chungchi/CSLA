clc; close all; clear all;
st=clock;
fprintf('begin computation...');
imgDir='./images/pair';
imgDir2='./images/pair/%s';
[Iset,listOfImages]=GetImagePair_demo(imgDir,imgDir2);

addpath(genpath('./External'));
addpath(genpath('./SACS_code'));
TIP11result = './cosmap11'; % TIP2011 result
TIP14result = './cosmap14'; % TIP2014 result
imgpath = './images/pair/';
gtpath = './cosdata/groundtruth'; % groundtruth path
global_mappath = './submaps/pair/old/globalsaliency'; % global saliency map path
local_mappath = './submaps/pair/old/localsaliency'; % local saliency map path
map_names = {'_Achanta.bmp','_Itti.bmp','_Hou.bmp','_color.bmp','_patch.bmp'};
map_num = length(map_names);
pridx = 0;
 
%% Parameter settings
for par_setting1 = -1 % beta1
for par_setting2 = -5 % alpha1
for par_setting3 = 0 % alpha2
for par_setting4 = 5:5:50 % k nearest neighbors

pridx = pridx + 1;
close all; clc;
par_setting1
par_setting2
par_setting3
par_setting4
for kk = 1:2:50
kk    
imglist = listOfImages(kk:kk+1);
imgset = Iset(kk:kk+1);       

img_num = length(imglist);
par.beta1 = 10^(par_setting1); % coefficient for pairwise term
par.alpha1 = 10^(par_setting2); % coefficient for second data term
par.alpha2 = 10^(par_setting3); % coefficient for third data term 
par.knn = par_setting4; % coefficient for k-nearest neightbor
par.largest_p_value = 1; % number of inter-image responses
par.ccodebook = 100; % how many bins in color histogram
par.cclusternum = 200; % how many run times in kmeans
par.clambda = 1; % affinity value, standard deviation denominator for color feature
par.glambda = 1; % affinity value, standard deviation denominator for saliency feature
par.numsupixel = 200; % superpixel size for regional fusion [use the same number with TIP2014]

%% Self-adaptive weight fusion
display('begin weight fusion')
inames = cell(1,img_num);
for i=1:img_num,
    inames{i} = strrep(imglist(i).name, '.bmp', ''); % replace string with another
end;

% read-in saliency maps
fnum = strrep(imglist(1).name, 'a.bmp', '');
fdir1 = sprintf('%s/%s',local_mappath,fnum);
fdir2 = sprintf('%s/%s',global_mappath,fnum);
Mset = cell(img_num, map_num);
for m=1:img_num,
    for n=1:map_num,
        if n<=3
        Mset{m,n} = imread([fdir1 '/' inames{m} map_names{n}]);
        else
        Mset{m,n} = imread([fdir2 '/' inames{m} map_names{n}]);
        end
    end; 
end;

% adaptive weight regional fusion for superpixels
k = par.knn;
numsupixel = par.numsupixel;
[w,supixels] = sacs_calWeight(map_names, inames, Mset, imgpath, k, numsupixel); 

% generate null saliency maps
saliency = cell(1,img_num);
for j=1:img_num,
    saliency{j} = zeros(size(Mset{j,1}));
end;

% gernerate saliency maps
for j = 1:img_num
%     N(j) = max(supixels{j}(:)); 
    temp_map = zeros(size(Mset{j,1}));
    for q = 1:max(supixels{j}(:))
        h = supixels{j} == q;
        for t = 1:map_num
        temp_map = temp_map + (w{j,q}(t)*h).*double(Mset{j,t});   
        s_avg{j,q}(t) = sum(sum(h.*double(Mset{j,t})))/sum(h(:)); % mean saliency for superpixels
        end
    end;
    saliency{j} = temp_map;
end;
 
% % save the result saliency maps to file
% for j=1:img_num,
%     rs = saliency{j};
%     rs = normalize(rs);
%     name = ['results/'  strrep(imglist(j).name, '.bmp' , '')];
%     fdir1 = sprintf('./%s_%s_%i.png',name,'fusion',k);
%     imwrite(rs,fdir1,'png');
% end;

%% Generating color and gabor features 
display('begin generating features')
cvector = []; % color vector
gvector = []; % gabor vector

for i=1:img_num,       
    IRGB = imread([imgpath '/' inames{i},'.bmp']);
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
%     gaborArray = gaborFilterBank(5,8,39,39);
%     featureVector_r = gaborFeatures(IRGB(:,:,1),gaborArray,1,1);
%     featureVector_g = gaborFeatures(IRGB(:,:,2),gaborArray,1,1);
%     featureVector_b = gaborFeatures(IRGB(:,:,3),gaborArray,1,1);
%     gvecc{i} = [featureVector_r,featureVector_g,featureVector_b];
    gvecc{i} = create_feature_space(IRGB); % gabor feature
    gvector = [gvector;gvecc{i}];
end

codebook_size = par.ccodebook;
cluster_maxnum = par.cclusternum;

% K-measn clustering for color features
des = cvector;
randid = floor(size(des,1)*rand(1,codebook_size))'+1;
randcen = des(randid,:);
[ccenv,ccenl] = do_kmeans(des', codebook_size, cluster_maxnum, randcen');
ccenl = ccenl+1;

% K-means clustering for gabor features
des = gvector;
randid = floor(size(des,1)*rand(1,codebook_size))'+1;
randcen = des(randid,:);
[gcenv,gcenl] = do_kmeans(des', codebook_size, cluster_maxnum, randcen');
gcenl = gcenl+1;

% generate the feature histogram for superpixels
for i=1:img_num, 
    if (i == 1)
      for j = 1:max(supixels{1}(:))    
          idx = find(supixels{i}(:) == j);
          cfeat{i,j} = hist(ccenl(idx),(1:codebook_size))/numel(idx);
          gfeat{i,j} = hist(gcenl(idx),(1:codebook_size))/numel(idx);
      end;
    elseif (i == 2)
      for j = 1:max(supixels{2}(:))     
          idx = find(supixels{i}(:) == j);
          cfeat{i,j} = hist(ccenl(idx+numel(supixels{1})),(1:codebook_size))/numel(idx);
          gfeat{i,j} = hist(gcenl(idx+numel(supixels{1})),(1:codebook_size))/numel(idx);
      end;   
    end       
end

%% Initial guess of A matrix for first data term
display('begin A matrix')
A = [];
for id = 1:img_num
   for i = 1:max(supixels{id}(:)),
    A_vec1 = (ones(1,map_num)-w{id,i}');
    A_vec2 = A_vec1./sum(A_vec1);
    A = vertcat(A,A_vec2);
   end
end

%% Calculating affinity matrix
display('begin affinity matrix')
ctotal = 0;
gtotal = 0;
num = 0;

for i = 1:img_num 
    for j = 1:max(supixels{i}(:))
        rx = find(supixels{i} == j);
        neigh = getneighbor_demo(supixels{i},rx);            
      for n = 1:neigh.num
          temp = neigh.ind(n);
          cdist = Ka2distance_demo(cfeat{i,j},cfeat{i,temp}); % color distance 
          gdist = Ka2distance_demo(gfeat{i,j},gfeat{i,temp}); % gabor distance
          ctotal = ctotal + cdist;
          gtotal = gtotal + gdist;
          num = num + 1;
          cdist = 0;
          gdist = 0;
      end
   end  
end

sigma_c = ctotal/num;
sigma_g = gtotal/num;

%
N(1) = max(supixels{1}(:));
N(2) = max(supixels{2}(:));
W{1} = zeros(N(1),N(1));
W{2} = zeros(N(2),N(2));
C{1} = zeros(N(1),N(2));
C{2} = zeros(N(2),N(1));

for i = 1:img_num  
    for j = 1:N(i)
        rx = find(supixels{i} == j);
        neigh = getneighbor_demo(supixels{i},rx); 
    
        for n = 1:neigh.num
        temp = neigh.ind(n);
        d1 = Ka2distance_demo(cfeat{i,j},cfeat{i,temp}); % color distance 
        d2 = Ka2distance_demo(gfeat{i,j},gfeat{i,temp}); % gabor distance
        W{i}(j,temp) = exp(-(par.clambda*(d1/sigma_c) + par.glambda*(d2/sigma_g)));
        end
        
        for k = 1: N(mod(i,2)+1)
        d3 = Ka2distance_demo(cfeat{i,j},cfeat{(mod(i,2)+1),k});
        d4 = Ka2distance_demo(gfeat{i,j},gfeat{(mod(i,2)+1),k});
        dist(k) = par.clambda*(d3/sigma_c) + par.glambda*(d4/sigma_g);            
        end
        [val,loc] = sort(dist(1:N(mod(i,2)+1)),'ascend');
        e{i,j} = exp(-val(1));
        e_candidate{i,j} = {j,loc(1)}; % check the matching superpixels across images
    
        for m = 1:par.largest_p_value
        C{i}(j,loc(m)) = exp(-val(m));
        end
        clear dist
    end        
end
% affinity matrix
M = [W{1},C{1};C{1}',W{2}];

for i = 1:img_num
   test_map = zeros(size(supixels{i}));
   for j = 1:N(i)
       hh = find(supixels{i} == j);
       zero_map = zeros(size(test_map));
       zero_map(hh) = 1;
       test_map = test_map + e{i,j}*zero_map;
   end
   test_e{i} = test_map;
end
    

figure;
subplot(2,2,1)
imname = [imgpath '/' [inames{1},'.bmp'] ];
img1 = im2double(imread(imname));
I_sp1 = segImage(img1,double(supixels{1}));
imshow(I_sp1);

subplot(2,2,2)
temp1 = test_e{1};
norm_emap1 = (temp1-min(temp1(:)))/(max(temp1(:)-min(temp1(:))));
imagesc(temp1);
colorbar
clear temp1

subplot(2,2,3)
imname = [imgpath '/' [inames{2},'.bmp'] ];
img2 = im2double(imread(imname));
I_sp2 = segImage(img2,double(supixels{2}));
imshow(I_sp2);

subplot(2,2,4)
temp2 = test_e{2};
norm_emap2 = (temp2-min(temp2(:)))/(max(temp2(:)-min(temp2(:))));
imagesc(temp2);
colorbar
clear temp2

AxesH = gcf;   % Not the GCF
F = getframe(AxesH);
fdir = sprintf('./%s/%s/%s.png','results/','emap',fnum);
imwrite(F.cdata,fdir,'png');

%% Calculating laplacian matrix for pairwise term
display('begin Laplacian matrix')
affinity = M;
clear D
% compute the degree matrix
for i=1:size(affinity,1)
    D(i,i) = sum(affinity(i,:));
end

L = D - affinity;
NL = D^(-1/2) * L * D^(-1/2);

%% Second data term
display('begin 2nd data term')
B = [];
for id = 1:img_num
   for i = 1:N(id),     
   temp_vec1 = (1-e{id,i}).*s_avg{id,i};
   temp_vec2 = exp(temp_vec1)./sum(exp(temp_vec1));
   B = vertcat(B,temp_vec2);
   end
end

%% Optimization
display('begin optimization')
L = NL; % assign normalized laplacian to optimization 
Nsum = N(1)+N(2);
clear x;
cvx_begin
    variable x(Nsum,map_num)
    minimize(trace(A*x') + par.alpha1*trace(B*x') + par.beta1*sum(quad_form(x(:,1),NL)+quad_form(x(:,2),NL)+quad_form(x(:,3),NL)+quad_form(x(:,4),NL)+quad_form(x(:,5),NL)) + ...
        par.alpha2*sum(quad_form(x(:,1),eye(Nsum))+quad_form(x(:,2),eye(Nsum))+quad_form(x(:,3),eye(Nsum))+quad_form(x(:,4),eye(Nsum))+quad_form(x(:,5),eye(Nsum))))             
    subject to 
    ones(1,map_num)*x' == ones(1,Nsum);
    0 <= x(:) <= 1
cvx_end

data1 = trace(A*x');
data2 = trace(B*x');
data3 = sum(quad_form(x(:,1),eye(Nsum))+quad_form(x(:,2),eye(Nsum))+quad_form(x(:,3),eye(Nsum))+quad_form(x(:,4),eye(Nsum))+quad_form(x(:,5),eye(Nsum)));
pair1 = sum(quad_form(x(:,1),NL)+quad_form(x(:,2),NL)+quad_form(x(:,3),NL)+quad_form(x(:,4),NL)+quad_form(x(:,5),NL));

%% Comparing pre(/post)-optimiztion weight
display('comparing pre(post) optimization weight')
for i = 1:img_num
   if i == 1
       for j = 1:N(1)
       w_new{i,j} = x(j,:);
       y(j,:) = w{i,j};
       end
   elseif i == 2
       for j = 1:N(2)
       w_new{i,j} = x(j+N(1),:);
       y(j+N(1),:) = w{i,j};
       end
   end    
end

%% Calculating the final region fusion maps
display('begin fusion maps generation')
for j=1:img_num,
    saliency{j} = zeros(size(Mset{j,1}));
end;
for j = 1:img_num
    spnum = max(supixels{j}(:)); 
    temp_map = zeros(size(Mset{j,1}));
    for q=1:spnum
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
    map = posdeal2(rs,0.4,6);%0.3 is used to control the threshold in equation 13
    % Note that, we use 0.4 for image pair saliency detection, and 0.3 for others. 
    map = normalize(map);
    ours{j} = im2double(map);    
    name = ['results/pairs/'  strrep(imglist(j).name, '.bmp' , '')];
    fdir1 = sprintf('./%s.png',name);
    imwrite(ours{j},fdir1,'png');
end;

gt = cell(img_num,1);% read-in groundtruth
TIP11 = cell(img_num,1);% read-in TIP2011 result
TIP14 = cell(img_num,1);% read-in TIP2014 result

for i=1:img_num,
        gt = im2double(imread([gtpath '/' inames{i} '.bmp'])); % read-in groundtruth
        tip11 = im2double(imread([TIP11result '/' inames{i} '.bmp'])); % read-in TIP2011 result
        tip14 = im2double(imread([TIP14result '/' inames{i} '.png'])); % read-in TIP2014 result
        
        if i == 1
        [prec11{kk},tpr11{kk}] = prec_rec(double(tip11(:))  ,gt(:),'plotROC',0,'plotPR',0,'numThresh',101);% Precision and Recall
        [prec14{kk},tpr14{kk}] = prec_rec(double(tip14(:))  ,gt(:),'plotROC',0,'plotPR',0,'numThresh',101);% Precision and Recall
        [precour{kk},tprour{kk}]=prec_rec(double(ours{i}(:)),gt(:),'plotROC',0,'plotPR',0,'numThresh',101);% Precision and Recall
        
        if kk == 1, 
        sum11 = zeros(numel(prec11{1},1));
        sum14 = zeros(numel(prec14{1},1));
        sumour{pridx}= zeros(numel(precour{1},1));
        tot11 = zeros(numel(tpr11{1},1));
        tot14 = zeros(numel(tpr14{1},1));
        totour{pridx}= zeros(numel(tprour{1},1));
        end
        
        s11  = prec11{kk}; sum11  = sum11 + s11';
        s14  = prec14{kk}; sum14  = sum14 + s14';
        sour = precour{kk};sumour{pridx} = sumour{pridx}+ sour';        
        t11  = tpr11{kk};  tot11  = tot11 + t11';
        t14  = tpr14{kk};  tot14  = tot14 + t14';
        tour = tprour{kk}; totour{pridx} = totour{pridx}+ tour';
        
        else
        [prec11{kk+1},tpr11{kk+1}] = prec_rec(double(tip11(:))   ,gt(:),'plotROC',0,'plotPR',0,'numThresh',101);% Precision and Recall
        [prec14{kk+1},tpr14{kk+1}] = prec_rec(double(tip14(:))   ,gt(:),'plotROC',0,'plotPR',0,'numThresh',101);% Precision and Recall
        [precour{kk+1},tprour{kk+1}]= prec_rec(double(ours{i}(:)),gt(:),'plotROC',0,'plotPR',0,'numThresh',101);% Precision and Recall 
        
        s11  = prec11{kk+1}; sum11  = sum11 + s11';
        s14  = prec14{kk+1}; sum14  = sum14 + s14';
        sour = precour{kk+1};sumour{pridx} = sumour{pridx}+ sour';
        t11  = tpr11{kk+1};  tot11  = tot11 + t11';
        t14  = tpr14{kk+1};  tot14  = tot14 + t14';
        tour = tprour{kk+1}; totour{pridx} = totour{pridx}+ tour';
        end
        
end;
end;
end;
end;
end;
end;
fprintf(' took %.2f minutes\n',etime(clock,st)/60);

%%
display('begin precision recall curve')
% % precision-recall curve
figure; hold on; xlabel('recall'); ylabel('precision');
plot([0, tot11/(kk+1)],[1 , sum11/(kk+1)], 'y'); % yellow
plot([0, tot14/(kk+1)],[1 , sum14/(kk+1)], 'm'); % magenta
plot([0, totour{1}/(kk+1)],[1 , sumour{1}/(kk+1)], 'c'); % cyan
% plot([0, totour{2}/(kk+1)],[1 , sumour{2}/(kk+1)], 'r'); % red
% plot([0, totour{3}/(kk+1)],[1 , sumour{3}/(kk+1)], 'g'); % green
% plot([0, totour{4}/(kk+1)],[1 , sumour{4}/(kk+1)], 'b'); % blue
% % plot([0. totour{5}/(kk+1)],[1 , sumour{5}/(kk+1)], 'k-*'); % white
% % plot([0. totour{6}/(kk+1)],[1 , sumour{6}/(kk+1)], 'k'); % black
legend('tip11','tip14','pridx1');%,'pridx2','pridx3','pridx4','pridx5','pridx6')
grid on; hold off;
% % attention****
% title('knn 5 10 15 20')
% % area under the precision-recall curve
% Int1 = trapz([0, tot11/(kk+1)], [1 , sum11/(kk+1)]);
% Int2 = trapz([0, tot14/(kk+1)], [1 , sum14/(kk+1)]);
% Int3 = trapz([0, totour{1}/(kk+1)],[1 , sumour{1}/(kk+1)]); % cyan
% Int4 = trapz([0, totour{2}/(kk+1)],[1 , sumour{2}/(kk+1)]); % red
% Int5 = trapz([0, totour{3}/(kk+1)],[1 , sumour{3}/(kk+1)]); % green
% Int6 = trapz([0, totour{4}/(kk+1)],[1 , sumour{4}/(kk+1)]); % blue
% Int7 = trapz([0. totour{5}/(kk+1)],[1 , sumour{5}/(kk+1)]); % white
% Int8 = trapz([0. totour{6}/(kk+1)],[1 , sumour{6}/(kk+1)]); % black



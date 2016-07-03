clc; close all; clear all;
%% Initial settings
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
%% Parameter settings
for kk = 1:2:210
display('change image pairs');
kk
pridx = 0;       
par.knn = 2^(0); % coefficient for k-nearest neightbor (1st data)
par.clambda = 1; % affinity value, standard deviation denominator for color feature (2nd data & pairwise)
par.glambda = 2; % affinity value, standard deviation denominator for saliency feature (2nd data & pairwise)
par.ccodebook = 100; % how many bins in color histogram (2nd data & pairwise)
par.cclusternum = 200; % how many run times in kmeans (2nd data & pairwise)
par.numsupixel = 200; % superpixel size for regional fusion [use the same number with TIP2014]

%% Self-adaptive weight fusion
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
saliency_tip14 = cell(1,img_num);
% for j=1:img_num,
%     saliency{j} = zeros(size(Mset{j,1}));
% end;

% gernerate saliency maps
for j = 1:img_num
    temp_map = zeros(size(Mset{j,1}));
    X = zeros(max(supixels{j}(:)),map_num);
    for q = 1:max(supixels{j}(:))
        h = supixels{j} == q;
        for t = 1:map_num
        temp_map = temp_map + (w{j,q}(t)*h).*double(Mset{j,t});   
        s_avg{j,q}(t) = sum(sum(h.*double(Mset{j,t})))/sum(h(:)); % mean saliency for superpixels
        X(q,t) = sum(sum(h.*double(Mset{j,t})))/sum(h(:)); % mean saliency for superpixels
        end
    end;
    saliency_tip14{j} = temp_map;
    lamda = 0.06;
    [~ ,E] = exact_alm_rpca(X,lamda);
    S{j} = double(E);
end;

saliency_accv14 = cell(1,img_num);

for j = 1:img_num
    spnum = max(supixels{j}(:)); 
    temp_map = zeros(size(Mset{j,1}));
    for q=1:spnum
        h = supixels{j} == q;
        temp_map = temp_map + (norm(S{j}(q,:)).*h);   
    end;
    saliency_accv14{j} = temp_map;
end;
 
% save the result saliency maps to file accv14
for j=1:img_num,
    rs = saliency_accv14{j};
    rs = (rs-min(rs(:)))/(max(rs(:))-min(rs(:)));
%     map = posdeal2(rs,0.4,6); % [TIP14] use 0.4 for image pair saliency detection, and 0.3 for others. 
    rs = normalize(rs);
    name = ['accv14/'  strrep(imglist(j).name, '.bmp' , '')];
    fdir3 = sprintf('./%s.png',name);
    imwrite(rs,fdir3,'png');
end;
end
%% Generating color and gabor features 
display('begin generating features')

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
for par_setting1 = -1 % beta1
for par_setting2 = -5% alpha1
for par_setting3 = 0 % alpha2
for par_setting4 = 0 % k nearest neighbors
pridx = pridx + 1;
close all; clc;
par_setting1
par_setting2
par_setting3
par_setting4
par.beta1 = 2^(par_setting1); % coefficient for pairwise term
par.alpha1 = 2^(par_setting2); % coefficient for second data term
par.alpha2 = 2^(par_setting3); % coefficient for third data term 
par.largest_p_value = 2^(par_setting4); % number of inter-image responses (pairwise)

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
e1 = [];
e2 = [];

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
        if i == 1,
           e1 = [e1,e{i,j}];
        else 
           e2 = [e2,e{i,j}];
        end
       
        for m = 1:par.largest_p_value
        C{i}(j,loc(m)) = exp(-val(m));
        end
        clear dist
    end        
end
affinity matrix
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
%     
% figure;
% subplot(2,2,1)
% imname = [imgDir '/' [inames{1},'.bmp'] ];
% img1 = im2double(imread(imname));
% I_sp1 = segImage(img1,double(supixels{1}));
% imshow(I_sp1);
% 
% subplot(2,2,2)
% emap1 = im2double(normalize(test_e{1}));
% imagesc(emap1);
% colorbar
% 
% subplot(2,2,3)
% imname = [imgDir '/' [inames{2},'.bmp'] ];
% img2 = im2double(imread(imname));
% I_sp2 = segImage(img2,double(supixels{2}));
% imshow(I_sp2);
% 
% subplot(2,2,4)
% emap2 = im2double(normalize(test_e{2}));
% imagesc(emap2);
% colorbar
% 
% AxesH = gcf;   % Not the GCF
% F = getframe(AxesH);
% fdir = sprintf('./%s/%s/%s.png','results/','emap',fnum);
% imwrite(F.cdata,fdir,'png');

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

e1 = (e1 - min(e1))/ (max(e1)-min(e1));
e2 = (e2 - min(e2))/ (max(e2)-min(e2));
B = [];
for id = 1:img_num
   for i = 1:N(id),
%       if id == 1
%       temp_vec1 = (1-e1(i)).*s_avg{id,i};
%       else
%       temp_vec1 = (1-e2(i)).*s_avg{id,i};
%       end
   temp_vec1 = (1 - e{id,i})*s_avg{id,i};
   temp_vec2 = exp(temp_vec1)./sum(exp(temp_vec1));
   B = vertcat(B,temp_vec2);
   end
end

%% Optimization
display('begin optimization')

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

%% Comparing pre(/post)-optimiztion weight
display('comparing pre(post) optimization weight')

for i = 1:img_num
   if i == 1
       for j = 1:max(supixels{1}(:))
       w_new{i,j} = x(j,:);
       y(j,:) = w{i,j};
       end
   elseif i == 2
       for j = 1:max(supixels{2}(:))
       w_new{i,j} = x(j+max(supixels{1}(:)),:);
       y(j+max(supixels{1}(:)),:) = w{i,j};
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
    map = posdeal2(rs,0.4,6); % [TIP14] use 0.4 for image pair saliency detection, and 0.3 for others. 
    map = normalize(map);
    ours{j} = im2double(map);    
    name = ['results/pairs/'  strrep(imglist(j).name, '.bmp' , '')];
    fdir4 = sprintf('./%s.bmp',name);
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
display('begin precision recall curve')
% precision-recall curve
figure; hold on; xlabel('recall'); ylabel('precision');
plot(fliplr([tot11{pridx}/((kk+1)),0]),fliplr([sum11{pridx}/((kk+1)),1]), 'y'); % yellow
plot(fliplr([tot14{pridx}/((kk+1)),0]),fliplr([sum14{pridx}/((kk+1)),1]), 'y-*'); % yellow
plot(fliplr([totour{1}/(kk+1),0]),fliplr([sumour{1}/(kk+1),1]), 'c'); % cyan
% plot(fliplr([totour{2}/(kk+1),0]),fliplr([sumour{2}/(kk+1),1]), 'c-*'); % cyan
% plot(fliplr([totour{3}/(kk+1),0]),fliplr([sumour{3}/(kk+1),1]), 'r'); % red
% plot(fliplr([totour{4}/(kk+1),0]),fliplr([sumour{4}/(kk+1),1]), 'r-*'); % red
% plot(fliplr([totour{5}/(kk+1),0]),fliplr([sumour{5}/(kk+1),1]), 'g'); % green
% plot(fliplr([totour{6}/(kk+1),0]),fliplr([sumour{6}/(kk+1),1]), 'g-*'); % black
% plot(fliplr([totour{7}/(kk+1),0]),fliplr([sumour{7}/(kk+1),1]), 'b'); % blue
% plot(fliplr([totour{8}/(kk+1),0]),fliplr([sumour{8}/(kk+1),1]), 'b-*'); % white
% plot(fliplr([totour{9}/(kk+1),0]),fliplr([sumour{9}/(kk+1),1]), 'k'); % black
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
% Int2 = trapz(fliplr([totour{2}/(kk+1),0]),fliplr([sumour{2}/(kk+1),1])); % red
% Int3 = trapz(fliplr([totour{3}/(kk+1),0]),fliplr([sumour{3}/(kk+1),1])); % green
% Int4 = trapz(fliplr([totour{4}/(kk+1),0]),fliplr([sumour{4}/(kk+1),1])); % blue
% Int5 = trapz(fliplr([totour{5}/(kk+1),0]),fliplr([sumour{5}/(kk+1),1])); % white
% Int6 = trapz(fliplr([totour{6}/(kk+1),0]),fliplr([sumour{6}/(kk+1),1])); % black
% Int7 = trapz(fliplr([totour{7}/(kk+1),0]),fliplr([sumour{7}/(kk+1),1])); % white
% Int8 = trapz(fliplr([totour{8}/(kk+1),0]),fliplr([sumour{8}/(kk+1),1])); % black
% Int9 = trapz(fliplr([totour{9}/(kk+1),0]),fliplr([sumour{9}/(kk+1),1])); % white
% Int10 = trapz(fliplr([totour{10}/(kk+1),0]),fliplr([sumour{10}/(kk+1),1])); % black
% Int11 = trapz(fliplr([totour{11}/(kk+1),0]),fliplr([sumour{11}/(kk+1),1])); % white
% Int12 = trapz(fliplr([totour{12}/(kk+1),0]),fliplr([sumour{12}/(kk+1),1])); % black
% jj = sprintf('%.4i/%.4i/%.4i/%.4i/%.4i/%.4i/%.4i/%.4i/%.4i/%.4i',Int1,Int2,Int3,Int4,Int5,Int6,Int7,Int8,Int9,Int10);
% title((jj))
AxesH = gcf;   % Not the GCF
F = getframe(AxesH);
fdir5 = sprintf('./%s/%s/%s.png','results','emap','auc');
imwrite(F.cdata,fdir5,'png');


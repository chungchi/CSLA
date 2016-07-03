clc; close all; clear all;
%% Initial setting
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
pridx = 0; st=clock;
%% Parameter settings
for m = -5:5
for n = 1    
for kk = 1:2:210
close all; clc;

kk    
pridx = pridx + 1;
par.clambda = 2^m; % affinity value, standard deviation denominator for color feature (2nd data & pairwise)
par.glambda = 2^n; % affinity value, standard deviation denominator for saliency feature (2nd data & pairwise)
par.ccodebook = 100; % how many bins in color histogram (2nd data & pairwise)
par.cclusternum = 200; % how many run times in kmeans (2nd data & pairwise)


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
 
% Generating color and gabor features 
display('begin generating features')

cvector = []; % color vector
gvector = []; % gabor vector

for i=1:img_num, 
    imname = [imgDir '/' [inames{i},'.bmp'] ];
    IRGB = imread(imname);       
    %------------generate the superpixels-------------------%
    superpixels = SLIC_mex( IRGB, 200, 20 );
    supixels{i} = superpixels;
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

% Calculating affinity matrix
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
N(1) = max(supixels{1}(:));
N(2) = max(supixels{2}(:));

for i = 1:img_num  
    for j = 1:N(i)                
        for k = 1: N(mod(i,2)+1)
        d3 = Ka2distance_demo(cfeat{i,j},cfeat{(mod(i,2)+1),k});
        d4 = Ka2distance_demo(gfeat{i,j},gfeat{(mod(i,2)+1),k});
        dist(k) = par.clambda*(d3/sigma_c) + par.glambda*(d4/sigma_g);            
        end
        [val,loc] = sort(dist(1:N(mod(i,2)+1)),'ascend');
        e{i,j} = exp(-val(1));
        e_candidate{i,j} = {j,loc(1)}; % check the matching superpixels across images   
        clear dist
    end        
end

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
imname = [imgDir '/' [inames{1},'.bmp'] ];
img1 = im2double(imread(imname));
I_sp1 = segImage(img1,double(supixels{1}));
imshow(I_sp1);

subplot(2,2,2)
emap1 = test_e{1};
norm_emap1 = (emap1-min(emap1(:)))/(max(emap1(:)-min(emap1(:))));
imagesc(norm_emap1);
colorbar
clear emap1

subplot(2,2,3)
imname = [imgDir '/' [inames{2},'.bmp'] ];
img2 = im2double(imread(imname));
I_sp2 = segImage(img2,double(supixels{2}));
imshow(I_sp2);

subplot(2,2,4)
emap2 = test_e{2};
norm_emap2 = (emap2-min(emap2(:)))/(max(emap2(:)-min(emap2(:))));
imagesc(emap2);
colorbar
clear emap2

AxesH = gcf;   % Not the GCF
F = getframe(AxesH);
fnum = strrep(imglist(1).name, 'a.bmp', '');
fdir = sprintf('./%s/%s/%s_%i_%i.png','results/','emap',fnum,m,n);
imwrite(F.cdata,fdir,'png');

gt = cell(img_num,1);% read-in groundtruth

for i=1:img_num,
     gt = im2double(imread([gtpath '/' inames{i} '.bmp'])); % read-in groundtruth
      
     if i == 1      
        [precour{kk},tprour{kk},~,thresh]=prec_rec(norm_emap1(:),gt(:),'plotROC',0,'plotPR',0,'numThresh',100);% Precision and Recall

        if kk == 1,
        sumour{pridx}= zeros(numel(precour{1},1)); % precision
        totour{pridx}= zeros(numel(tprour{1},1)); % recall
        end
        
        sour = precour{kk}; sumour{pridx} = sumour{pridx}+ sour';  % sum precision       
        tour = tprour{kk}; totour{pridx} = totour{pridx}+ tour'; % total recall
        
     else
        [precour{kk+1},tprour{kk+1},~,thresh]= prec_rec(norm_emap2(:),gt(:),'plotROC',0,'plotPR',0,'numThresh',100);% Precision and Recall 
        
        sour = precour{kk+1};sumour{pridx} = sumour{pridx}+ sour'; % sum precision
        tour = tprour{kk+1}; totour{pridx} = totour{pridx}+ tour'; % total recall
     end
        
end;

end;
end;
end;

%%
display('begin precision recall curve')
% precision-recall curve
figure; hold on; xlabel('recall'); ylabel('precision');
plot(fliplr([totour{1}/(kk+1),0]),fliplr([sumour{1}/(kk+1),1]), 'c'); % cyan
plot(fliplr([totour{2}/(kk+1),0]),fliplr([sumour{2}/(kk+1),1]), 'r'); % red
plot(fliplr([totour{3}/(kk+1),0]),fliplr([sumour{3}/(kk+1),1]), 'g'); % green
plot(fliplr([totour{4}/(kk+1),0]),fliplr([sumour{4}/(kk+1),1]), 'b'); % blue
plot(fliplr([totour{5}/(kk+1),0]),fliplr([sumour{5}/(kk+1),1]), 'b-*'); % white
plot(fliplr([totour{6}/(kk+1),0]),fliplr([sumour{6}/(kk+1),1]), 'k'); % black
plot(fliplr([totour{7}/(kk+1),0]),fliplr([sumour{7}/(kk+1),1]), 'k-*'); % white
plot(fliplr([totour{8}/(kk+1),0]),fliplr([sumour{8}/(kk+1),1]), 'g-*'); % black
plot(fliplr([totour{9}/(kk+1),0]),fliplr([sumour{9}/(kk+1),1]), 'y'); % yellow
plot(fliplr([totour{10}/(kk+1),0]),fliplr([sumour{10}/(kk+1),1]), 'm'); % magenta
plot(fliplr([totour{11}/(kk+1),0]),fliplr([sumour{11}/(kk+1),1]), 'm'); % magenta
legend('pridx1','pridx2','pridx3','pridx4','pridx5','pridx6','pridx7','pridx8','pridx9','pridx10','pridx11')
grid on; hold off;
% attention****
% area under the precision-recall curve
Int1 = trapz(fliplr([totour{1}/(kk+1),0]),fliplr([sumour{1}/(kk+1),1]));
Int2 = trapz(fliplr([totour{2}/(kk+1),0]),fliplr([sumour{2}/(kk+1),1]));
Int3 = trapz(fliplr([totour{3}/(kk+1),0]),fliplr([sumour{3}/(kk+1),1])); % cyan
Int4 = trapz(fliplr([totour{4}/(kk+1),0]),fliplr([sumour{4}/(kk+1),1])); % red
Int5 = trapz(fliplr([totour{5}/(kk+1),0]),fliplr([sumour{5}/(kk+1),1])); % green
Int6 = trapz(fliplr([totour{6}/(kk+1),0]),fliplr([sumour{6}/(kk+1),1])); % blue
Int7 = trapz(fliplr([totour{7}/(kk+1),0]),fliplr([sumour{7}/(kk+1),1])); % white
Int8 = trapz(fliplr([totour{8}/(kk+1),0]),fliplr([sumour{8}/(kk+1),1])); % black
Int9 = trapz(fliplr([totour{9}/(kk+1),0]),fliplr([sumour{9}/(kk+1),1])); % white
Int10 = trapz(fliplr([totour{10}/(kk+1),0]),fliplr([sumour{10}/(kk+1),1])); % black
Int11 = trapz(fliplr([totour{11}/(kk+1),0]),fliplr([sumour{11}/(kk+1),1])); % black
% jj = sprintf('%.4i_%.4i_%.4i_%.4i_%.4i_%.4i',Int1,Int2,Int3,Int4,Int5,Int6);
% title((jj))
% AxesH = gcf;   % Not the GCF
% F = getframe(AxesH);
% fdir5 = sprintf('./%s/%s/%s.png','results','emap','auc');
% imwrite(F.cdata,fdir,'png');


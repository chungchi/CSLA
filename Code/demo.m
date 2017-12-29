% ICASSP17: Image Cosaliency Detection via Locally Adaptive Saliency Map Fusion
% Code Author: Chung-Chi "Charles" Tsai
% Email: chungchi@tamu.edu
% Date: April 2017
% Note: Please download the "CVX" and get the license file from "http://cvxr.com/cvx/download"
% If you think this code is useful, please consider citing:
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% @inproceedings{tsai2017image,
%   title={Image co-saliency detection via locally adaptive saliency map fusion},
%   author={Tsai, Chung-Chi, Qian, Xiaoning and Lin, Yen-Yu},
%   booktitle={Acoustics, Speech and Signal Processing (ICASSP), 2017 IEEE International Conference on},
%   pages={1897--1901},
%   year={2017},
%   organization={IEEE}}
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% initial settings
clc; close all; clear all;
addpath(genpath('cvx-64')); cvx_setup;
addpath(genpath('SLIC'));
addpath(genpath('External'));
addpath(genpath('Eval'));

dataSet = 'pair105';
gt.Dir = './cosdata/gt/pair/pair105'; % input gt path
fdir.main ='./cosdata/images/pair/pair105'; % input image path
fdir.sub = dir([fdir.main, '/']); % input image path
fdir.sub = fdir.sub(~ismember({fdir.sub.name},{'.','..','.DS_Store'}));
fileNum = length(fdir.sub);
map.folder = './submaps/pair/pair105'; % local saliency maps path
map.names = {'_Achanta.bmp','_Itti.bmp','_Hou.bmp','_color.bmp','_patch.bmp'};
map.num = length(map.names);

par.knn = 50; % coefficient for k-nearest neightbor (1st data)
par.thes = 0.1; % threshold for the noisy saliency value (1st data)
par.neighbors = 1; % number of inter-image responses (pairwise)
par.clambda = 1; % affinity value, standard deviation denominator for color feature (2nd data & pairwise)
par.glambda = 0.5; % affinity value, standard deviation denominator for saliency feature (2nd data & pairwise)
par.numsupixel = 200; % superpixel size for regional fusion [use the same number with SACS-TIP2014]
par.ccodebook = 100; % how many bins in color histogram (1st data & pairwise)
par.cclusternum = 150; % how many clusters in the K-means clustering

for fnum = 1:fileNum
    
    fprintf('begin processing image pairs: %s\r', fdir.sub(fnum).name);
    pair_num = fdir.sub(fnum).name;
    
    img.Dir1 = ([fdir.main,'/',(pair_num)]);
    img.Dir2 = [img.Dir1,'/','*.bmp']; % input image path
    img.list = dir(img.Dir2);
    IRGB{1} = (imread(sprintf([img.Dir1,'/%s'],img.list(1).name)));
    IRGB{2} = (imread(sprintf([img.Dir1,'/%s'],img.list(2).name)));
    img.num = length(IRGB);
    
    gt.Dir1 = ([gt.Dir,'/',(pair_num)]);
    gt.Dir2 = [gt.Dir1,'/','*.bmp'];
    gt.list = dir(gt.Dir2);
    MaskList{1} = (imread(sprintf([gt.Dir1,'/%s'],gt.list(1).name)));
    MaskList{2} = (imread(sprintf([gt.Dir1,'/%s'],gt.list(2).name)));
    
    % clean testing data storage place
    testpath = sprintf('test/pair/%s/%s/%s',...
        dataSet,'icassp17',fdir.sub(fnum).name);
    mkdir(testpath);
    
    whichfolder1 = testpath;
    dinfo1 = dir(fullfile(whichfolder1,'*.*'));
    for K = 1 : length(dinfo1)
        thisfile = fullfile(whichfolder1, dinfo1(K).name);
        delete(thisfile);
    end
    
    % generating color and gabor features
    disp('begin generating features...');
    cvector = []; % color vector
    gvector = []; % gabor vector
    
    clear cvecc gvecc;
    for i = 1:img.num
        if strfind(img.list(i).name,'.bmp')
            img.name{i} = strrep(img.list(i).name,'.bmp','');
        elseif strfind(img.list(i).name,'.png')
            img.name{i} = strrep(img.list(i).name,'.png','');
        elseif strfind(img.list(i).name,'.jpg')
            img.name{i} = strrep(img.list(i).name,'.jpg','');
        elseif strfind(img.list(i).name,'.JPG')
            img.name{i} = strrep(img.list(i).name,'.JPG','');
        end
        RGB = im2double(IRGB{i});
        R = RGB(:,:,1);
        G = RGB(:,:,2);
        B = RGB(:,:,3);
        rgb = [R(:),G(:),B(:)];
        Ycbr = double(rgb2ycbcr(IRGB{i}));
        Y = Ycbr(:,:,1)/255;
        Cb = Ycbr(:,:,2)/255;
        Cr = Ycbr(:,:,3)/255;
        ybr = [Y(:),Cb(:),Cr(:)];
        [Cl,Ca,Cb] = rgb2lab(IRGB{i}(:,:,1),IRGB{i}(:,:,2),IRGB{i}(:,:,3));
        Lab = [Cl(:)/100,(Ca(:)+110)/220,(Cb(:)+110)/220];
        cvecc{i} = [rgb,Lab,ybr]; % color feature
        cvector = [cvector;cvecc{i}];
        gvecc{i} = create_feature_space(IRGB{i}); % gabor feature
        gvector = [gvector;gvecc{i}];
    end
    
    % read-in saliency maps
    submap_dir = sprintf('%s/%s',map.folder,fdir.sub(fnum).name); % local maps path
    Mset = read_maps(img.name,img.num,map.names,map.num,submap_dir);
    
    % first unary term
    disp('begin A matrix');
    [w,supixels] = sacs_accv14_thes(map.names,img.name,Mset,img.Dir1,par);
    A = set_A_new(2,supixels,w,map.num);
    
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
    for i=1:img.num
        if (i == 1)
            for j = 1:max(supixels{1}(:))
                idx = find(supixels{i}(:) == j);
                cfeat{i,j} = hist(ccenl(idx),(1:par.ccodebook))/numel(idx);
                gfeat{i,j} = hist(gcenl(idx),(1:par.ccodebook))/numel(idx);
            end
        elseif (i == 2)
            for j = 1:max(supixels{2}(:))
                idx = find(supixels{i}(:) == j);
                cfeat{i,j} = hist(ccenl(idx+numel(supixels{1})),(1:par.ccodebook))/numel(idx);
                gfeat{i,j} = hist(gcenl(idx+numel(supixels{1})),(1:par.ccodebook))/numel(idx);
            end
        end
    end
    
    % affinity matrix
    disp('begin affinity matrix');
    [par.sigma_c,par.sigma_g] = compute_sigma_new(img.num,supixels,cfeat,gfeat);% Computer the normalization constant
    [M,e] = set_affinity(supixels,img.num,cfeat,gfeat,par);% compute the affinity matrix
    
    % calculating laplacian matrix for pairwise term
    affinity = M;
    clear D
    
    % compute the degree matrix
    for i=1:size(affinity,1)
        D(i,i) = sum(affinity(i,:));
    end
    L = D - affinity;
    [V,E,W] = eig(L);
    m = W*E^(1/2);
    
    % second unary term
    disp('begin 2nd data term');
    s_avg = avg_sm(supixels, 2, map.num, Mset);
    B = set_B(e,s_avg,2,supixels);
    
    % optimization
    pridx = 0;
    for setting1 = 3
        for setting2 = 2
            for setting3 = 0
                
                pridx = pridx + 1;
                par.alpha1 = 2^(setting1);
                par.alpha2 = 2^(setting2);
                par.beta1  = 2^(setting3);
                
                % optimization
                disp('begin optimization...');
                Nsum = size(A,1);
                clear x sm;
                cvx_begin quiet
                variable x(Nsum,map.num)
                for k = 1:map.num
                    sm(1,k) = quad_form(m'*x(:,k),eye(Nsum));
                end
                minimize(par.alpha1*trace(A*x')+par.alpha2*trace(B*x')+par.beta1*sum(sm)+sum_square(x(:)))
                subject to
                ones(1,map.num)*x' == ones(1,Nsum);
                0 <= x(:) <= 1;
                cvx_end
                
                time(fnum,pridx) = cvx_cputime;
                fprintf('cvx computation time = %s\n',num2str(time(fnum,pridx)));
                
                % optimiztion weight
                disp('comparing pre(post) optimization weight');
                w_new = set_w(2, supixels, x);
                
                % cosaliency maps
                disp('begin fusion maps generation')
                
                for j = 1:2
                    temp_map = zeros(size(Mset{j,1}));
                    for q=1:max(supixels{j}(:))
                        h = supixels{j} == q;
                        for t = 1:map.num
                            temp_map = temp_map + (w_new{j,q}(t)*h).*double(Mset{j,t});
                        end
                    end
                    rs = normalize(temp_map);
                    SalMap{j}  = normalize(posdeal2(rs,0.4,6));
                    name = sprintf('%s/%s_%s.png',testpath,img.name{j},num2str(pridx));
                    imwrite(SalMap{j},name);
                end
                [MAE(fnum),AP(fnum),AUC(fnum),mF(fnum),wf(fnum)] = EvalSalImage(SalMap,MaskList);
            end
        end
    end
end


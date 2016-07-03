function call_gabor
close all;

%%read in input images
im=imread('./images/pair/amira/amira1.bmp');
im1=imresize(im,1);

im2=imread('./images/pair/amira/amira2.bmp');
im2=imresize(im2,1);

%create the texture feature space for each image
feaImg1=create_feature_space(im1);
feaImg2=create_feature_space(im2);
[n r c]=size(im1);

%the following code is for visualization 
%for one point in image 1, see how similar it is w.r.t. the feature
%representation to *all* points in image 2. This is shown as a colormapped
%image

%click on one point and read it in as [x y]
imshow(im1); title('click a point on this image');

[x y]=ginput(1);
x=round(x); y=round(y); hold on;
plot(x,y,'ro')
hold off;

ind1=sub2ind([n r],x,y);

%find the corresponding feature vect
Fea_temp=repmat(feaImg1(ind1,:),n*r,1);
size(Fea_temp)

%construct the rbf kernel for this vector and the entire feature space
%of other image & display
params.gamma=300;
% W=rbf_kernel_matrix(Fea_temp',feaImg2',params);
W = abs(Fea_temp' - feaImg2');
%W = W/max(max(W));
%W=W(1,:);
W=sum(W,1);
W=reshape(W,[n r]);
figure, imshow(im2);
figure,imagesc(imresize(double(W),2));
colormap(Jet)
colorbar
%figure,imshow(imresize(im2,2));


%%-------------------------------------------------------------------------
%function to create feature space for image
function feaImg=create_feature_space(im)

[n r c] = size(im);

%choice of bandwidth and orientation
%f = [0,2];
%f is 1/lambda where lambda is the wavelength (pixels). Valid numbers are
%between 2 and 256
lambda = [3];
f = 1/lambda;
%theta = [0 pi/6 pi/3 3*pi/4];
theta = 0:pi/8:(pi-pi/8);

feaImg=[];
for i=1:length(f)
    for j=1:length(theta)
        gamma = 0;
        [G Gimg_r]=gaborfilterVikas(im(:,:,1),2,2,f(i),theta(j),gamma);
        [G Gimg_g]=gaborfilterVikas(im(:,:,2),2,2,f(i),theta(j),gamma);
        [G Gimg_b]=gaborfilterVikas(im(:,:,3),2,2,f(i),theta(j),gamma);
        Gimg=[];
        Gimg=cat(3,cat(3,Gimg_r,Gimg_g),Gimg_b);
        imshow(uint8(Gimg));
        Gimg=reshape(Gimg,n*r,c);
        feaImg=[feaImg Gimg];
    end
end

for i=1:length(f)
    for j=1:length(theta)
        gamma = -0.5*pi;
        [G Gimg_r]=gaborfilterVikas(im(:,:,1),2,2,f(i),theta(j),gamma);
        [G Gimg_g]=gaborfilterVikas(im(:,:,2),2,2,f(i),theta(j),gamma);
        [G Gimg_b]=gaborfilterVikas(im(:,:,3),2,2,f(i),theta(j),gamma);
        Gimg=[];
        Gimg=cat(3,cat(3,Gimg_r,Gimg_g),Gimg_b);
        imshow(uint8(Gimg));
        Gimg=reshape(Gimg,n*r,c);
        feaImg=[feaImg Gimg];
    end
end

size(feaImg)

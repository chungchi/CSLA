function feaImg=create_feature_space(im)

[n r c] = size(im);

% choice of bandwidth and orientation
% f = [0,2];
% f is 1/lambda where lambda is the wavelength (pixels). Valid numbers are
% between 2 and 256
lambda = [3,5,7];
f = 1./lambda;
% theta = [0 pi/6 pi/3 3*pi/4];
theta = 0:pi/8:(pi-pi/8);

feaImg=[];
for i=1:length(f)
    for j=1:length(theta)
        gamma = 0;
        [G Gimg_r]=gaborfilterVikas(im(:,:,1),f(i),theta(j),gamma); % [gabor filter, R channel output] 
        [G Gimg_g]=gaborfilterVikas(im(:,:,2),f(i),theta(j),gamma); % [gabor filter, G channel output] 
        [G Gimg_b]=gaborfilterVikas(im(:,:,3),f(i),theta(j),gamma); % [gabor filter, B channel output] 
        Gimg=[];
        Gimg=cat(3,cat(3,Gimg_r,Gimg_g),Gimg_b); % contatenate arrays along 3rd dimension
        Gimg=reshape(Gimg,n*r,c);
        feaImg=[feaImg Gimg];
    end
end

for i=1:length(f)
    for j=1:length(theta)
        gamma = -0.5*pi;
        [G Gimg_r]=gaborfilterVikas(im(:,:,1),f(i),theta(j),gamma); % [gabor filter, R channel output] 
        [G Gimg_g]=gaborfilterVikas(im(:,:,2),f(i),theta(j),gamma); % [gabor filter, G channel output] 
        [G Gimg_b]=gaborfilterVikas(im(:,:,3),f(i),theta(j),gamma); % [gabor filter, B channel output] 
        Gimg=[];
        Gimg=cat(3,cat(3,Gimg_r,Gimg_g),Gimg_b); % contatenate arrays along 3rd dimension
        Gimg=reshape(Gimg,n*r,c);
        feaImg=[feaImg Gimg];
    end
end

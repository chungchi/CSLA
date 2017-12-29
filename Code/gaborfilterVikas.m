function [G,gabout] = gaborfilterVikas(I,f,theta,phi,~)

if isa(I,'double')~=1 
    I = double(I);
end

%width
sigmaX = (1/f)*0.56;
sigmaY = (1/f)*0.56;

%ellipticity
gamma = 0.5;

for x = -fix(sigmaX):fix(sigmaX)
    for y = -fix(sigmaY):fix(sigmaY)
        xp = x * cos(theta) + y * sin(theta);
        yp = y * cos(theta) - x * sin(theta);
        
        indx = fix(sigmaX)+x+1;
        indy = fix(sigmaY)+y+1;
        G(indx,indy) = exp(-0.5*((xp/sigmaX)^2 + gamma^2*(yp/sigmaY)^2))*cos(2*pi*f*xp + phi);
               
    end
end

Imgabout = conv2(I,double(imag(G)),'same');
Regabout = conv2(I,double(real(G)),'same');

gabout = sqrt(Imgabout.*Imgabout + Regabout.*Regabout);
gabout = gabout./max(max(gabout));
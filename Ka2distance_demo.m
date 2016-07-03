function out=Ka2distance_demo(His1,His2)
c1=(His1-His2).^2;
c2=His1+His2;
nz=find(c2~=0);
out=sum(c1(nz)./c2(nz));

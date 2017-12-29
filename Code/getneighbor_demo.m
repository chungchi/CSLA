function neigh=getneighbor_demo(segmask,rx)
se= strel('square',3);
zmask=zeros(size(segmask));
zmask(rx)=1;
zmaskd=imdilate(zmask,se);
xmask=zmaskd-zmask;
u=find(xmask==1);
if isempty(u)
    neigh.num=0;
    neigh.wt=0;
    neigh.ind=0;    
else
    ring=segmask(u);
    rd=unique(ring);

    neigh.num=length(rd);
    neigh.ind=rd;
    for j=1:length(rd)
        neigh.wt(j)=length(find(ring==rd(j)))/length(u);
    end
end
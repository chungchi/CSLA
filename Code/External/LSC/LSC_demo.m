% You are free to use, change or redistribute this code for any
% non-commrecial purposes.If you use this software,please cite the
% following in any resulting publication and email us:
% [1] Zhengqin Li, Jiansheng Chen, Superpixel Segmentation using Linear 
%     Spectral Clustering, IEEE Conference on Computer Vision and Pattern 
%     Recognition (CVPR), Jun. 2015 
% (C) Zhengqin Li, Jiansheng Chen, 2014
% li-zq12@mails.tsinghua.edu.cn
% jschenthu@mail.tsinghua.edu.cn
% Tsinghua University
clear all;clc;close all;

% First layer superpixels by Linear Spectral Clustering
name='01';
img=imread([name,'.jpg']);
gaus=fspecial('gaussian',3);
I=imfilter(img,gaus);
[nRows,nCols,dims] = size(I);
superpixelNum=100;
ratio=0.075;

label=LSC_mex(I,superpixelNum,ratio);
DisplaySuperpixel(label,img,name);
DisplayLabel(label,name);
%% Find the region representation for each superpixel
% convert RGB to CieLab
R = double(I(:,:,1));
G = double(I(:,:,2));
B = double(I(:,:,3));
[L,a,b] = RGB2Lab(R,G,B);

% get the number of superpixel for the first layer
Num_1 = length(unique(label));

% initial setting
Color = 20;
Distance = Color*ratio;
ColNum = sqrt(superpixelNum*nCols/nRows);
RowNum = superpixelNum/ColNum;
StepX = nRows/RowNum;
StepY = nCols/ColNum;

sigmaL1 = 0;
sigmaL2 = 0;
sigmaa1 = 0;
sigmaa2 = 0;
sigmab1 = 0;
sigmab2 = 0;
sigmax1 = 0;
sigmax2 = 0;
sigmay1 = 0;
sigmay2 = 0;

% get the region representation for each superpixel
for i = 1:nRows
    
    for j = 1:nCols     
        
        thetaL=(L(i,j)/255)*pi/2;
        thetaa=(a(i,j)/255)*pi/2;
        thetab=(b(i,j)/255)*pi/2;
        thetax=(i/StepX)*pi/2;
        thetay=(j/StepY)*pi/2;
        L1(i,j) = Color*cos(thetaL);
        L2(i,j) = Color*sin(thetaL);
	    a1(i,j) = Color*cos(thetaa)*2.55;
	    a2(i,j) = Color*sin(thetaa)*2.55;
	    b1(i,j) = Color*cos(thetab)*2.55;
	    b2(i,j) = Color*sin(thetab)*2.55;
	    x1(i,j) = Distance*cos(thetax);
	    x2(i,j) = Distance*sin(thetax);
	    y1(i,j) = Distance*cos(thetay);
	    y2(i,j) = Distance*sin(thetay);
                
        sigmaL1 = sigmaL1 + L1(i,j)/(nCols*nRows);
        sigmaL2 = sigmaL2 + L2(i,j)/(nCols*nRows);
        sigmaa1 = sigmaa1 + a1(i,j)/(nCols*nRows);
        sigmaa2 = sigmaa2 + a2(i,j)/(nCols*nRows);
        sigmab1 = sigmab1 + b1(i,j)/(nCols*nRows);
        sigmab2 = sigmab2 + b2(i,j)/(nCols*nRows);
        sigmax1 = sigmax1 + x1(i,j)/(nCols*nRows);
        sigmax2 = sigmax2 + x2(i,j)/(nCols*nRows);
        sigmay1 = sigmay1 + y1(i,j)/(nCols*nRows);
        sigmay2 = sigmay2 + y2(i,j)/(nCols*nRows);
               
    end
    
end

W = zeros(nRows,nCols);

for i=1:nRows
	
    for j=1:nCols		
		 
        W(i,j)=L1(i,j)*sigmaL1+L2(i,j)*sigmaL2+...
		       a1(i,j)*sigmaa1+a2(i,j)*sigmaa2+...
			   b1(i,j)*sigmab1+b2(i,j)*sigmab2+...
			   x1(i,j)*sigmax1+x2(i,j)*sigmax2+...
			   y1(i,j)*sigmay1+y2(i,j)*sigmay2;
			
               L1(i,j) = L1(i,j)/W(i,j);
               L2(i,j) = L2(i,j)/W(i,j);
			   a1(i,j) = a1(i,j)/W(i,j);
			   a2(i,j) = a2(i,j)/W(i,j);
			   b1(i,j) = b1(i,j)/W(i,j);
			   b2(i,j) = b2(i,j)/W(i,j);
			   x1(i,j) = x1(i,j)/W(i,j);
			   x2(i,j) = x2(i,j)/W(i,j);
			   y1(i,j) = y1(i,j)/W(i,j);
			   y2(i,j) = y2(i,j)/W(i,j);
   
    end
    
end
 
for i = 1:Num_1
      [x,y] = find(label == i);
      Count = length(x);

      centerL1(i) = 0;
      centerL2(i) = 0;
      centera1(i) = 0;
      centera2(i) = 0;
      centerb1(i) = 0;
      centerb2(i) = 0;
      centerx1(i) = 0;
      centerx2(i) = 0;
      centery1(i) = 0;
      centery2(i) = 0;
      locationx(i) = 0;
      locationy(i) = 0;
     
      for j = 1:Count  
                    
          centerL1(i) = centerL1(i) + L1(x(j),y(j))/Count;
          centerL2(i) = centerL2(i) + L2(x(j),y(j))/Count;
          centera1(i) = centera1(i) + a1(x(j),y(j))/Count;
          centera2(i) = centera2(i) + a2(x(j),y(j))/Count;
          centerb1(i) = centerb1(i) + b1(x(j),y(j))/Count;
          centerb2(i) = centerb2(i) + b2(x(j),y(j))/Count;
          centerx1(i) = centerx1(i) + x1(x(j),y(j))/Count;
          centerx2(i) = centerx2(i) + x2(x(j),y(j))/Count;
          centery1(i) = centery1(i) + y1(x(j),y(j))/Count;
          centery2(i) = centery2(i) + y2(x(j),y(j))/Count;
          locationx(i) = locationx(i) + x(j)/Count;
          locationy(i) = locationy(i) + y(j)/Count;
      
      end
      
end

%% Find seeds for second-layer linear spectral clustering
% initial setting
seedNum =  100;
ColNum2 = floor(sqrt(seedNum*nCols/nRows));
RowNum2 = floor(seedNum/ColNum2);
Row_step = floor(nRows/RowNum2);
Col_step = floor(nCols/ColNum2);
Row_remain = nRows-Row_step*RowNum2;
Col_remain = nCols-Col_step*ColNum2;
t1=0;
count = 1;
centerx = 0; 
centery = 0;

for i=1:RowNum2
	
		t2=0;
		for j=1:ColNum2
		
			centerx = (i-1)*Row_step+0.5*Row_step+t1;
			centery = (j-1)*Col_step+0.5*Col_step+t2;
			
            if (centerx>=nRows-1)
                centerx = nRows-1;
            end
            
            if (centery >= nCols-1)
                centery = nCols-1;
            end
            
            if(t2 < Col_remain)
				t2 = t2+1;
            end
            
            point_array(count).x = centerx;
            point_array(count).y = centery;
			count = count+1;
        end
		
		if (t1 < Row_remain)
				t1 = t1+1;
        end
           
end
%% Create the centroids 

superpixelNum2 = RowNum2*ColNum2;

for i=1:superpixelNum2

		centroidL1(i) = 0;
		centroidL2(i) = 0;
		centroida1(i) = 0;
		centroida2(i) = 0;
		centroidb1(i) = 0;
		centroidb2(i) = 0;
		centroidx1(i) = 0;
		centroidx2(i) = 0;
		centroidy1(i) = 0;
		centroidy2(i) = 0;
		x = point_array(i).x;
		y = point_array(i).y;
		
        if (x-Row_step <= 0)
            minX = 0;
        else
            minX = x-Row_step;
        end
        
		if (y-Col_step<=0)
            minY = 0;
        else
            minY = y-Col_step;
        end
        
		if (x+Row_step>=nRows-1)
            maxX = nRows-1;
        else
            maxX = x+Row_step;
        end
        
		if (y+Col_step>=nCols-1)
            maxY = nCols-1;
        else
            maxY = y+Col_step;
        end
        

        log1 = (minX <= locationx);
        log2 = (locationx <= maxX);
        log3 = (minY <= locationy);
        log4 = (locationy <= maxY);
        idx = find(log1 & log2 & log3 & log4);
		count = numel(idx);
        
        for j = 1:count
            centroidL1(i) = centroidL1(i) + centerL1(idx(j))/count;
            centroidL2(i) = centroidL2(i) + centerL2(idx(j))/count;
            centroida1(i) = centroida1(i) + centera1(idx(j))/count;
            centroida2(i) = centroida2(i) + centera2(idx(j))/count;
            centroidb1(i) = centroidb1(i) + centerb1(idx(j))/count;
            centroidb2(i) = centroidb2(i) + centerb2(idx(j))/count;
            centroidx1(i) = centroidx1(i) + centerx1(idx(j))/count;
            centroidx2(i) = centroidx2(i) + centerx2(idx(j))/count;
            centroidy1(i) = centroidy1(i) + centery1(idx(j))/count;
            centroidy2(i) = centroidy2(i) + centery2(idx(j))/count;
        end

end

%% K-means clustering
iterationNum = 20;

for (iteration=1:iterationNum)
		
    dist = 10^10*ones(1,superpixelNum2);
  
    for (i=1:superpixelNum2)
		
        x = point_array(i).x;
		y = point_array(i).y;
		
        if (x-Row_step <= 0)
            minX = 0;
        else
            minX = x-Row_step;
        end
        
		if (y-Col_step<=0)
            minY = 0;
        else
            minY = y-Col_step;
        end
        
		if (x+Row_step>=nRows-1)
            maxX = nRows-1;
        else
            maxX = x+Row_step;
        end
        
		if (y+Col_step>=nCols-1)
            maxY = nCols-1;
        else
            maxY = y+Col_step;
        end
        
        log1 = (minX <= locationx);
        log2 = (locationx <= maxX);
        log3 = (minY <= locationy);
        log4 = (locationy <= maxY);
        idx = find(log1 & log2 & log3 & log4);
		count = numel(idx);
        
        for j = 1:count
            D = (centerL1(j)-centriodL1(i))*(centerL1(j)-centerL1[i])+
			(L2[m][n]-centerL2[i])*(L2[m][n]-centerL2[i])+
			(a1[m][n]-centera1[i])*(a1[m][n]-centera1[i])+
			(a2[m][n]-centera2[i])*(a2[m][n]-centera2[i])+
			(b1[m][n]-centerb1[i])*(b1[m][n]-centerb1[i])+
			(b2[m][n]-centerb2[i])*(b2[m][n]-centerb2[i])+
			(x1[m][n]-centerx1[i])*(x1[m][n]-centerx1[i])+
			(x2[m][n]-centerx2[i])*(x2[m][n]-centerx2[i])+
			(y1[m][n]-centery1[i])*(y1[m][n]-centery1[i])+
			(y2[m][n]-centery2[i])*(y2[m][n]-centery2[i]);
		
        if(D<dist[m][n])
					
			label[m*nCols+n]=i;
			dist[m][n]=D;
       
        end
        
        end					
    end
  

 for(int i=0;i<seedNum;i++)
		
			centerL1[i]=0;
			centerL2[i]=0;
			centera1[i]=0;
			centera2[i]=0;
			centerb1[i]=0;
			centerb2[i]=0;
			centerx1[i]=0;
			centerx2[i]=0;
			centery1[i]=0;
			centery2[i]=0;
			WSum[i]=0;
			clusterSize[i]=0;
			seedArray[i].x=0;
			seedArray[i].y=0;
        end
        



		for(int i=0;i<nRows;i++)
		
			for(int j=0;j<nCols;j++)
			
				int L=label[i*nCols+j];
				double Weight=W[i][j];
				centerL1[L]+=Weight*L1[i][j];
				centerL2[L]+=Weight*L2[i][j];
				centera1[L]+=Weight*a1[i][j];
				centera2[L]+=Weight*a2[i][j];
				centerb1[L]+=Weight*b1[i][j];
				centerb2[L]+=Weight*b2[i][j];
				centerx1[L]+=Weight*x1[i][j];
				centerx2[L]+=Weight*x2[i][j];
				centery1[L]+=Weight*y1[i][j];
				centery2[L]+=Weight*y2[i][j];
				clusterSize[L]++;
				WSum[L]+=Weight;
				seedArray[L].x+=i;
				seedArray[L].y+=j;
            end   
        end
        
        for(int i=0;i<seedNum;i++)
		
			WSum[i]=(WSum[i]==0)?1:WSum[i];
			clusterSize[i]=(clusterSize[i]==0)?1:clusterSize[i];
        end
		for(int i=0;i<seedNum;i++)
		
			centerL1[i]/=WSum[i];
			centerL2[i]/=WSum[i];
			centera1[i]/=WSum[i];
			centera2[i]/=WSum[i];
			centerb1[i]/=WSum[i];
			centerb2[i]/=WSum[i];
			centerx1[i]/=WSum[i];
			centerx2[i]/=WSum[i];
			centery1[i]/=WSum[i];
			centery2[i]/=WSum[i];
			seedArray[i].x/=clusterSize[i];
			seedArray[i].y/=clusterSize[i];
        end
        
    end

   

	




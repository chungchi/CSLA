function [Iset,listOfImages]=GetImagePair_demo(imgDir,imgDir2)
oldPwd = pwd;
cd(imgDir);
x = dir;
listOfImages = [];
for i = 1:length(x),
    if x(i).isdir == 0 
        listOfImages = [listOfImages; x(i)];
    end;
end;
cd(oldPwd);

fid=imgDir2;
for j = 1:length(listOfImages)
    fileName = listOfImages(j).name;
    rfid=sprintf(fid,fileName);
    Irgb=imread(rfid);
    Iset{j}=Irgb;
end

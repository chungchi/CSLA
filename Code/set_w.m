function w_new = set_w(img_num, supixels, x)
addnum = 0;

for i = 1:img_num
    %    if i == 1
    %        for j = 1:max(supixels{1}(:))
    %        w_new{i,j} = x(j,:);
    % %        y(j,:) = w{i,j};
    %        end
    %    elseif i == 2
    %        for j = 1:max(supixels{2}(:))
    %        w_new{i,j} = x(j+max(supixels{1}(:)),:);
    % %        y(j+max(supixels{1}(:)),:) = w{i,j};
    %        end
    %    end
    for j = 1:max(supixels{i}(:))
        w_new{i,j} = x(j+addnum,:);
        %        y(j,:) = w{i,j};
    end
    addnum = addnum + max(supixels{i}(:));
    
    
end

function B = set_B(e, s_avg, img_num, supixels)
% e1 = (e1 - min(e1))/ (max(e1)-min(e1));
% e2 = (e2 - min(e2))/ (max(e2)-min(e2));
B = [];
for id = 1:img_num
    e_val = (e{id}-repmat(min(e{id}),size(e{id},1),1))./repmat(max(e{id})-min(e{id}),size(e{id},1),1);
    for i = 1:max(supixels{id}(:)),
        mean_val = mean(e_val(i,:));
        std_val = std(e{id}(i,:));
        men = heaviside(mean_val-std_val);
        %       if id == 1
        %       temp_vec1 = (1-e1(i)).*s_avg{id,i} + e1(i).*(1-s_avg{id,i});
        %       else
        %       temp_vec1 = (1-e2(i)).*s_avg{id,i} + e2(i).*(1-s_avg{id,i});
        %       end
        %         temp_vec2 = temp_vec1;
        
        %       temp_vec1 = (1 - e{id,i})*s_avg{id,i};
        temp_vec1 = (1-men).*s_avg{id,i}+men.*(1-s_avg{id,i});
        temp_vec2 = exp(temp_vec1)./sum(exp(temp_vec1));
        %       temp_vec2 = (temp_vec1)./sum((temp_vec1));
        B = vertcat(B,temp_vec2);
    end
end
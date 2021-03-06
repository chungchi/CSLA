function B = set_B(e, s_avg, img_num, supixels)
B = [];
for id = 1:img_num
    e_val = (e{id}-repmat(min(e{id}),size(e{id},1),1))./repmat(max(e{id})-min(e{id}),size(e{id},1),1);
    for i = 1:max(supixels{id}(:)),
        mean_val = mean(e_val(i,:));
        std_val = std(e{id}(i,:));
        men = heaviside(mean_val-std_val);
        temp_vec1 = (1-men).*s_avg{id,i}+men.*(1-s_avg{id,i});
        temp_vec2 = exp(temp_vec1)./sum(exp(temp_vec1));
        B = vertcat(B,temp_vec2);
    end
end
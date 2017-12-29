function A = set_A_new(img_num,supixels,w,map_num)
A = [];
for id = 1:img_num,
   for i = 1:max(supixels{id}(:)), 
    A_vec1 = ones(1,map_num)-w{id,i}';   
    A_vec2 = exp(A_vec1)./sum(exp(A_vec1));
    A = vertcat(A,A_vec2);
   end;
end;
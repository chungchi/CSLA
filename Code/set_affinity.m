function [M,e] = set_affinity(supixels,img_num,cfeat,gfeat,par)

for i = 1:img_num,
    iii = 0;
    for ii = 1:img_num,
        N1 = max(supixels{i}(:));
        N2 = max(supixels{ii}(:));
%         e{i} = zeros(N1,img_num-1);
        W{i,ii} = zeros(N1,N2);
        if (i == ii)
            for j = 1:N1,
                rx = find(supixels{i} == j);
                neigh = getneighbor_demo(supixels{i},rx);
                for n = 1:neigh.num,
                    temp = neigh.ind(n);
                    d1 = Ka2distance_demo(cfeat{i,j},cfeat{ii,temp}); % color distance
                    d2 = Ka2distance_demo(gfeat{i,j},gfeat{ii,temp}); % gabor distance
                    W{i,ii}(j,temp) = exp(-(par.clambda*(d1/par.sigma_c) + par.glambda*(d2/par.sigma_g)));
                end;
            end;
        else
            iii = iii+1;
            for j = 1:N1,
                for k = 1:N2,
                    d3 = Ka2distance_demo(cfeat{i,j},cfeat{ii,k});
                    d4 = Ka2distance_demo(gfeat{i,j},gfeat{ii,k});
                    dist(k) = par.clambda*(d3/par.sigma_c) + par.glambda*(d4/par.sigma_g);
                end;
                [val,loc] = sort(dist,'ascend');
                for m = 1:min(par.neighbors,N2),
                    W{i,ii}(j,loc(m)) = exp(-val(m));                    
                end;
                e{i}(j,iii) = exp(-val(m));
                clear dist
            end;
        end;
    end;
end;

% affinity matrix
M = cell2mat(W);
M = sqrt(M.*M');

% N(1) = max(supixels{1}(:));
% N(2) = max(supixels{2}(:));
% W{1} = zeros(N(1),N(1));
% W{2} = zeros(N(2),N(2));
% C{1} = zeros(N(1),N(2));
% C{2} = zeros(N(2),N(1));
% e1 = [];
% e2 = [];
% 
% for i = 1:img_num,
%     N(i) = max(supixels{i}(:));
%     for j = 1:N(i),
%         W{i} = zeros(max(N(i),N(i)));
%         rx = find(supixels{i} == j);
%         neigh = getneighbor_demo(supixels{i},rx); 
%     
%         for n = 1:neigh.num,
%         temp = neigh.ind(n);
%         d1 = Ka2distance_demo(cfeat{i,j},cfeat{i,temp}); % color distance 
%         d2 = Ka2distance_demo(gfeat{i,j},gfeat{i,temp}); % gabor distance
%         W{i}(j,temp) = exp(-(par.clambda*(d1/par.sigma_c) + par.glambda*(d2/par.sigma_g)));
%         end;
%         
%         for k = 1:N(mod(i,2)+1),
%         d3 = Ka2distance_demo(cfeat{i,j},cfeat{(mod(i,2)+1),k});
%         d4 = Ka2distance_demo(gfeat{i,j},gfeat{(mod(i,2)+1),k});
%         dist(k) = par.clambda*(d3/par.sigma_c) + par.glambda*(d4/par.sigma_g);            
%         end;
%         [val,loc] = sort(dist(1:N(mod(i,2)+1)),'ascend');
%         e{i,j} = exp(-val(1));
%         e_candidate{i,j} = {j,loc(1)}; % check the matching superpixels across images
%         if i == 1,
%            e1 = [e1,e{i,j}];
%         else 
%            e2 = [e2,e{i,j}];
%         end;
%        
%         for m = 1:min(par.largest_p_value,N(mod(i,2)+1)),
%         C{i}(j,loc(m)) = exp(-val(m));
%         end;
%         clear dist
%     end;        
% end;

% affinity matrix
% m = sqrt(C{1}.*C{2}');
% M = [W{1},m;m',W{2}];

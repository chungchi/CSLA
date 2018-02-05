function [M,e] = set_affinity(supixels,img_num,cfeat,gfeat,par)

for i = 1:img_num,
    iii = 0;
    for ii = 1:img_num,
        N1 = max(supixels{i}(:));
        N2 = max(supixels{ii}(:));
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


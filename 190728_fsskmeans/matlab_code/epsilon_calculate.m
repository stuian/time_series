function [ eps ] = epsilon_calculate( MinPts,aDTW_data,chose_eps )
%   calculate epsilon for DBSCAN
n = size(aDTW_data,1);
k_dis = zeros(1,n);
for i = 1:n
    temp = aDTW_data(i,:);
    for j = 1:MinPts
        [k_dis(i),indx] = min(temp);
        temp(indx) = inf;
    end
end
k_dis_sort = sort(k_dis);
eps = k_dis_sort(chose_eps);
end


% 给定DTW矩阵，使用k-dist方法找出合适的Eps参数
load('data\yoga.mat');
dataset = yoga;
n = length(dataset);
dis_matrix = zeros(n,n);
for i = 1:n-1
    for j = i+1:n
%        dis_matrix(i,j) = tDTW_calculate(dataset{i}.data,dataset{j}.data);
%        dis_matrix(j,i) = dis_matrix(i,j);
         dis_matrix(i,j) = ub_ED(dataset{i}.data,dataset{j}.data);%使用ED试试
         dis_matrix(j,i) = dis_matrix(i,j);
    end
end
k = 5;
k_dist = zeros(1,n);
dis_matrix_sorted = sort(dis_matrix);
for i = 1:n
    k_dist(i) = dis_matrix_sorted(k+1,i);
end

k_dist_sort = sort(k_dist,'descend');





function dis_matrix = Dis_Matrix(train_data, matrixType)
% train_data: 聚类数据集
% matrixType: 计算的矩阵种类
% output: dis_matrix = 计算得到的data间距离矩阵。
n = length(train_data);
dis_matrix = zeros(n,n);
% matrixType 为1时计算所有data之间的tDTW矩阵
if matrixType == 1
    for i = 1:n
        for t = 1:n
            dis_matrix(i,t) = tDTW_calculate(train_data{i}.data, train_data{t}.data);
        end
    end
elseif matrixType == 2
    for i = 1:n
        for t = 1:n
            dis_matrix(i,t) = ub_ED(train_data{i}.data, train_data{t}.data);
        end
    end
end
end
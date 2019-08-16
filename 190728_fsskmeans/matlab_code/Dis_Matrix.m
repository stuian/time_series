function dis_matrix = Dis_Matrix(train_data, matrixType)
% train_data: �������ݼ�
% matrixType: ����ľ�������
% output: dis_matrix = ����õ���data��������
n = length(train_data);
dis_matrix = zeros(n,n);
% matrixType Ϊ1ʱ��������data֮���tDTW����
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
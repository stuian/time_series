function [indx,RandIndex_tDTW] = K_means_trueDTW(C,train_data,label)
%K_MEANS_TRUEDTW
%ʹ��tDTW���kmeans�ľ����������¼�������ݼ��ϵ�RandIndex������ʱ�䡣
k = length(C);
n = length(train_data);
iter = 15;
indx = zeros(1,n);

%ʹ��tDTW_calculate�����������tDTW���󣬺������㷨��������������С�
tDTW = zeros(k,n);
for i = 1:k
    for t = 1:n
        tDTW(i,t) = tDTW_calculate(C{i}.data, train_data{t}.data);
    end
end

for i = 1:iter    %50 iterations
%find nearest center point and assign a data point
    for p = 1:n
        temp = inf;
        for q = 1:k
            if tDTW(q,p) < temp
                temp = tDTW(q,p);
                indx(p) = q;
            end
        end
    end
    
    %update Center point
	for k1 = 1:k
		indx1 = find(indx == k1);
		temp = zeros(size(C{1}));
		for k2 = 1:length(indx1)
			temp = temp + train_data{indx1(k2)}.data; %���б�עΪk1�ĵ��data��ֵ�ĺ�
		end
		temp = temp/length(indx1);
		C{k1}.data = temp;
    end
    %recalculate tDTW
    for s = 1:k
        for t = 1:n
             tDTW(s,t) = tDTW_calculate(C{s}.data, train_data{t}.data);
        end
    end
end
    RandIndex_tDTW = RandIndex(indx,label);
end


function [indx,RandIndex_tDTW] = K_means_trueDTW_auxi(C,train_data,label,MustLink,CannotLink)
%K_MEANS_TRUEDTW
%全部使用tDTW完成kmeans的聚类分析，记录各个数据集合的RandIndex和运行时间。
k = length(C);
n = length(train_data);
iter = 15;
indx = zeros(1,n); % 所有标识初始化为0

%使用tDTW_calculate函数，计算出tDTW矩阵，后续的算法会基于这个矩阵进行。
% tDTW = zeros(k,n);
% for i = 1:k
%     for t = 1:n
%         tDTW(i,t) = tDTW_calculate(C{i}.data, train_data{t}.data);
%     end
% end

% 记录分配到各个类中的data
classes = {};
for i = 1:k
    classes{i} = [];
end

for i = 1:iter    %iterations
    %calculate tDTW
    tDTW = zeros(k,n);
    for s = 1:k
        for t = 1:n
             tDTW(s,t) = tDTW_calculate(C{s}.data, train_data{t}.data);
        end
    end
    %find nearest center point and assign a data point
    for p = 1:n
        temp = inf;
        for q = 1:k
            if tDTW(q,p) < temp %这一段代码都很奇怪？
                %加入CannotLink条件
                for s = 1:length(CannotLink)
                    if p == CannotLink{s}(1) && ismember(CannotLink{s}(2),classes{q})...
                        || p == CannotLink{s}(2) && ismember(CannotLink{s}(1),classes{q})
                        continue
                    else
                        temp = tDTW(q,p);
                        indx(p) = q;
                % 加入MustLink条件
                        for u = 1:length(MustLink)
                            if p == MustLink{u}(1)
                                indx(MustLink{u}(2)) = q;
                            elseif p == MustLink{u}(2)
                                indx(MustLink{u}(1)) = q;
                            end
                        end
                    end
                end
            end
        end
    end
    
    %update Center point
    for k1 = 1:k
		indx1 = find(indx == k1);
		temp = zeros(size(C{1}));
		for k2 = 1:length(indx1)
			temp = temp + train_data{indx1(k2)}.data; %所有标注为k1的点的data的和
		end
		temp = temp/length(indx1);
		C{k1}.data = temp;
    end

end
    RandIndex_tDTW = RandIndex(indx,label);
end


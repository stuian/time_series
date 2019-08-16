function [indx,RandIndex_tDTW] = K_means_trueDTW_auxi(C,train_data,label,MustLink,CannotLink)
%K_MEANS_TRUEDTW
%ȫ��ʹ��tDTW���kmeans�ľ����������¼�������ݼ��ϵ�RandIndex������ʱ�䡣
k = length(C);
n = length(train_data);
iter = 15;
indx = zeros(1,n); % ���б�ʶ��ʼ��Ϊ0

%ʹ��tDTW_calculate�����������tDTW���󣬺������㷨��������������С�
% tDTW = zeros(k,n);
% for i = 1:k
%     for t = 1:n
%         tDTW(i,t) = tDTW_calculate(C{i}.data, train_data{t}.data);
%     end
% end

% ��¼���䵽�������е�data
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
            if tDTW(q,p) < temp %��һ�δ��붼����֣�
                %����CannotLink����
                for s = 1:length(CannotLink)
                    if p == CannotLink{s}(1) && ismember(CannotLink{s}(2),classes{q})...
                        || p == CannotLink{s}(2) && ismember(CannotLink{s}(1),classes{q})
                        continue
                    else
                        temp = tDTW(q,p);
                        indx(p) = q;
                % ����MustLink����
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
			temp = temp + train_data{indx1(k2)}.data; %���б�עΪk1�ĵ��data�ĺ�
		end
		temp = temp/length(indx1);
		C{k1}.data = temp;
    end

end
    RandIndex_tDTW = RandIndex(indx,label);
end


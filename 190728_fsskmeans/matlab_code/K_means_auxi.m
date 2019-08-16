function [indx,RandIndex_aDTW] = K_means_auxi(C,train_data,beta, label,needSigma,useBound, MustLink,CannotLink)
iter = 15;
k = length(C);
n = length(train_data);
if useBound == 0
    aDTW = aDTW_calculate(C,train_data,beta);
else
    aDTW = Bound_calculate(C, train_data,useBound);
end
indx = zeros(1,n);

if needSigma == 1
    aDTW_std = std(aDTW(:));
    sigma = sigmoid(aDTW_std)*aDTW_std;
    sigma = sigma*2;
else
    sigma = 0;
end

% 记录分配到各个类中的data
classes = {};
for i = 1:k
    classes{i} = []; %后面没有更新？
end

for i = 1:iter    %iter deside how many iterations
% 	aDTW = aDTW_update(C,train_data,aDTW,sigma);    %update
    %find nearest center point
    for p = 1:n
        [min1, indx1, min2, indx2] = findTwoMin(aDTW, p);
        if abs(min2 - min1) >= sigma
            % add CannotLink
            for s = 1:length(CannotLink)
                if p == CannotLink{s}(1) && ismember(CannotLink{s}(2),classes{indx1})...
                    || p == CannotLink{s}(2) && ismember(CannotLink{s}(1),classes{indx1})
                    indx(p) = indx2;
                else
                    indx(p) = indx1;
                end
            end
        else
            min1 = tDTW_calculate(C{indx1}.data, train_data{p}.data);
            min2 = tDTW_calculate(C{indx2}.data, train_data{p}.data);
            aDTW(indx1,p) = min1;
            aDTW(indx2,p) = min2;
            if min1<min2
                %add CannotLink
                for s = 1:length(CannotLink)
                    if p == CannotLink{s}(1) && ismember(CannotLink{s}(2),classes{indx1})...
                        || p == CannotLink{s}(2) && ismember(CannotLink{s}(1),classes{indx1})
                        indx(p) = indx2;
                    else
                        indx(p) = indx1;
                    end
                end
            else
                for s = 1:length(CannotLink)
                    if p == CannotLink{s}(1) && ismember(CannotLink{s}(2),classes{indx2})...
                        || p == CannotLink{s}(2) && ismember(CannotLink{s}(1),classes{indx2})
                        indx(p) = indx1;
                    else
                        indx(p) = indx2;
                    end
                end
            end
        end
        %add MustLink
        for u = 1:length(MustLink)
            if p == MustLink{u}(1)
                indx(MustLink{u}(2)) = indx(p);
            elseif p == MustLink{u}(2)
                indx(MustLink{u}(1)) = indx(p);
            end
        end
    end


	%update Center point
	for k1 = 1:k
		indx1 = find(indx == k1);
		temp = zeros(size(C{1}));
		for k2 = 1:length(indx1)
			temp = temp + train_data{indx1(k2)}.data;    %所有标注为k1的点的data的和
		end
		temp = temp/length(indx1);
		C{k1}.data = temp;
	end
	%calculate aDTW again
	aDTW = aDTW_calculate(C,train_data,beta);
%     RandIndex_aDTW(i) = RandIndex(indx,label);
%     fprintf('第%d次递归，RandIndex为：%f\n',i,RandIndex_aDTW(i))
end
RandIndex_aDTW = RandIndex(indx,label);
end

function [min1,indx1, min2, indx2] = findTwoMin(aDTW, p)
    curr_colum = aDTW(:,p);
    [min1,indx1] = min(curr_colum);
    curr_colum(indx1) = inf;
    [min2,indx2] = min(curr_colum);
end

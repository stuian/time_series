%RIGHT
function aDTW = aDTW_calculate(C,D,beita)%C are center points
%calculate aDTW
k = length(C);
n = length(D);
aDTW = zeros(k,n);
for i = 1:k
    for j = i:n
        LB = lb_keogh(C{i}.data,D{j}.data);
        UB = ub_ED(C{i}.data,D{j}.data);
        aDTW(i,j) = LB + beita*(UB - LB);
    end
end
end

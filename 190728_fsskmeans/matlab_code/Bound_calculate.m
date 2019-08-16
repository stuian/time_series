function bound = Bound_calculate(C,D,UorL)
k = length(C);
n = length(D);
bound = zeros(k,n);
if UorL == 1 % calculate upper bounds
    bound = UB_calculate(C,D);
    a = 1; %?
elseif UorL == 2 %calculate lower bounds
    bound = LB_calculate(C,D);
end
end

function UB = UB_calculate(C,D)
k = length(C);
n = length(D);
UB = zeros(k,n);
for i = 1:k
    for j = 1:n
        UB(i,j) = ub_ED(C{i}.data,D{j}.data);
    end
end
end

function LB = LB_calculate(C,D)
k = length(C);
n = length(D);
LB = zeros(k,n);
for i = 1:k
    for j = i+1:n %?
        LB(i,j) = lb_keogh(C{i}.data,D{j}.data);
    end
end
end

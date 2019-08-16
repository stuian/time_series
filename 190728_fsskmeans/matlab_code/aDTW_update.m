%RIGHT
function aDTW = aDTW_update(C,D,aDTW,sigma)
k = length(C);
n = length(D);
aDTW_flag = zeros(k,n);
%update aDTW
for t = 1:n%line
	for i = 1:k%row
		for j = (i+1):k
			if abs(aDTW(i,t)-aDTW(j,t)) < sigma
				if aDTW_flag(i,t)==0
					aDTW(i,t) = tDTW_calculate(C{i}.data,D{t}.data);%update
                    aDTW(j,t) = tDTW_calculate(C{j}.data,D{t}.data);
					aDTW_flag(i,t) = 1;
				end
			end
		end
	end
end
end
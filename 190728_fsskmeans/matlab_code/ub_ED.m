function ub = ub_ED(data1,data2)
	ub=sqrt(sum(sum((data1-data2).^2)));
end
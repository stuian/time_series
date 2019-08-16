function lb=lb_keogh(data1,data2)

r=ceil(length(data1)/7);
mintemp=data1;
maxtemp=data1;
for k1=1:size(data1,2)
    temp=data1(:,k1);
    for k2=1:size(data1,1)
        mintemp(k2,k1)=min(temp(max(k2-r,1):min(k2+r,length(temp))));
        maxtemp(k2,k1)=max(temp(max(k2-r,1):min(k2+r,length(temp))));
    end
end
lb=sqrt(sum(sum([(data2>maxtemp).*(data2-maxtemp);(data2<mintemp).*(mintemp-data2)].^2)));
end
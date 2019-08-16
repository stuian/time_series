%RIGHT
function tDTW = tDTW_calculate(data1,data2)
%calculate tDTW between data1 and data2
n = size(data1, 1);
m = size(data2, 1);
% ¼ÆËãÖ¡Æ¥Åä¾àÀë¾ØÕó
d = zeros(n,m);
for i = 1:n
    for j = 1:m
        d(i,j) = sum((data1(i,:)-data2(j,:)).^2);
    end
end
% ÀÛ»ý¾àÀë¾ØÕó
D = ones(n,m) * realmax;
D(1,1) = d(1,1);
% ¶¯Ì¬¹æ»®
for i = 2:n
    for j = 1:m
        D1 = D(i-1,j);
        if j>1
            D2 = D(i-1,j-1);
        else
            D2 = realmax;
        end
        if j>2
            D3 = D(i-1,j-2);
        else
            D3 = realmax;
        end
        D(i,j) = d(i,j) + min([D1,D2,D3]);
    end
end
tDTW = D(n,m);
end
%     len=length(data1);
%     gamma=zeros(len+1,len+1);
%     gamma(1,:)=inf;
%     gamma(:,1)=inf;
%     gamma(1,1)=0;
%     for i=2:len+1
%         for j=2:len+1
%             gamma(i,j)=(sum((data1(i-1,:)-data2(j-1,:)).^2)) + min([gamma(i-1,j-1),gamma(i-1,j),gamma(i,j-1)]);
%         end
%     end
%     tDTW = sqrt(gamma(len+1,len+1));
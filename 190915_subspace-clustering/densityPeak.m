function [centerLabel,dc_w] = densityPeak( K )
%UNTITLED 寻找 K 个密度峰值
%   densityPeak( distence_between, N, K )
%   distence_between(N,N,R)
    global single_distence_between;
    [~,N,R] = size(single_distence_between);
    density = zeros(1,N);   %密度
    sigma_value = zeros(1,N); %密度大于序列n的序列到它距离最小值 
    sigma_label = zeros(1,N); %相应的序列号
    distence_w = zeros(N,N);  %两两序列之间带权距离

    for n=1:N  %每个序列依次计算带权距离
        for m=n:N 
            for r=1:R
                distence_w(n,m) = distence_w(n,m) +  (1/R) * single_distence_between(n,m,r)^2;
            end
            distence_w(n,m) = sqrt(distence_w(n,m));
            distence_w(m,n) = distence_w(n,m);
        end
    end

    percent = 0.02; %小于截断距离的数据平均占2%
    temp_sort = sort(distence_w,2);
    dc_w = mean(temp_sort(:,round(N*percent))); %计算截断距离


    for n=1:N   %每个序列依次计算局部密度
        [row,~] = find(distence_w(n,:) < dc_w);     %距离小于截断距离
        [~,density(1,n)] = size(row);  %局部密度
    end

    [~,density_label] = sort(density,'descend');  %按照密度从大到小分配剩下的序列
    for n=1:N
        if n==1    %最高值情况
            [temp_value,temp_label] = max(distence_w(density_label(n),:));  
            sigma_label(1,density_label(n)) = temp_label;
            sigma_value(1,density_label(n)) = temp_value;
        else
            bigger_label = density_label(1:n-1);
            [temp_value,temp_label] = min(distence_w(density_label(n),bigger_label));
            sigma_label(1,density_label(n)) = bigger_label(temp_label);
            sigma_value(1,density_label(n)) = temp_value;
        end          
    end                        

    u = sigma_value.*density;
    [~,sort_label] = sort(u,'descend'); %按照density*sigma排序
    centerLabel = zeros(1,K);
    for k=1:K
        centerLabel(k) = sort_label(k);
    end

end
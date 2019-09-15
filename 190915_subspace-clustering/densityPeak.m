function [centerLabel,dc_w] = densityPeak( K )
%UNTITLED Ѱ�� K ���ܶȷ�ֵ
%   densityPeak( distence_between, N, K )
%   distence_between(N,N,R)
    global single_distence_between;
    [~,N,R] = size(single_distence_between);
    density = zeros(1,N);   %�ܶ�
    sigma_value = zeros(1,N); %�ܶȴ�������n�����е���������Сֵ 
    sigma_label = zeros(1,N); %��Ӧ�����к�
    distence_w = zeros(N,N);  %��������֮���Ȩ����

    for n=1:N  %ÿ���������μ����Ȩ����
        for m=n:N 
            for r=1:R
                distence_w(n,m) = distence_w(n,m) +  (1/R) * single_distence_between(n,m,r)^2;
            end
            distence_w(n,m) = sqrt(distence_w(n,m));
            distence_w(m,n) = distence_w(n,m);
        end
    end

    percent = 0.02; %С�ڽضϾ��������ƽ��ռ2%
    temp_sort = sort(distence_w,2);
    dc_w = mean(temp_sort(:,round(N*percent))); %����ضϾ���


    for n=1:N   %ÿ���������μ���ֲ��ܶ�
        [row,~] = find(distence_w(n,:) < dc_w);     %����С�ڽضϾ���
        [~,density(1,n)] = size(row);  %�ֲ��ܶ�
    end

    [~,density_label] = sort(density,'descend');  %�����ܶȴӴ�С����ʣ�µ�����
    for n=1:N
        if n==1    %���ֵ���
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
    [~,sort_label] = sort(u,'descend'); %����density*sigma����
    centerLabel = zeros(1,K);
    for k=1:K
        centerLabel(k) = sort_label(k);
    end

end
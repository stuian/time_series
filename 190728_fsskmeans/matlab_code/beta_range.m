clear;clc;
file_path='data';
file_dir=dir(file_path);
file_name={'Adiac','ECG5000','Earthquakes','FaceAll','MALLAT',...
    'MedicalImages','POC','Strawberry','wafer','yoga'};

xx=0.05:0.05:0.95;
for k = 1:length(file_name)
    train_path=strcat(file_path,'\',file_name(k));
    train_data=struct2cell(load(train_path{1}));
    train_data = train_data{1};
    n = length(train_data);
    tDTW_data = zeros(n,n);
    for i = 1:n-1
        for j = i+1:n
            tDTW_data(i,j) = tDTW_calculate(train_data{i}.data,train_data{j}.data);
            tDTW_data(j,i) = tDTW_data(i,j);
        end
    end
    cnt = 1;
    error = zeros(1,18);
    for beta = 0.05:0.05:0.95
        aDTW_data = aDTW_calculate(train_data,train_data,beta);
        err = abs(aDTW_data - tDTW_data)./tDTW_data;
        for kk1=1:n
            for kk2=1:n
                if(isnan(err(kk1,kk2))) %剔除无穷大的情况（分母为0）
                    err(kk1,kk2)=0;
                end
            end
        end
        error(cnt) = sum(sum(err))/n/n;
        cnt = cnt+1;
    end
    plot(xx,error,'-.')
    hold on
    axis([0,1,0,1]);
    xlabel('\beta');
    ylabel('Error');
    fprintf('已经执行到数据集%d\n',k);
end

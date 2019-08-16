clear;clc;
time_scale_aDTW = zeros(3,100);
time_scale_tDTW = zeros(3,100);

% load('data\wafer.mat');
% cnt = 1;
% for m = 1:50
%     test_data_temp = wafer(1:m);
%     tic
%     [~,RI_aDTW(1)] = main(test_data_temp,0);%0 stand for using aDTW
%     time_scale_aDTW(1,cnt) = toc;
%     tic
%     [~,RI_tDTW(1)] = main(test_data_temp,1);%1 stand for using tDTW
%     time_scale_tDTW(1,cnt) = toc;
%     cnt = cnt+1;
% end
% disp('data_scale1 has done!');
% 
% 
% 
% load('data\yoga.mat');
% cnt = 1;
% for m = 1:50
%     test_data_temp = yoga(1:m);
%     tic
%     [~,RI_aDTW(1)] = main(test_data_temp,0);%0 stand for using tDTW
%     time_scale_aDTW(2,cnt) = toc;
%     tic
%     [~,RI_tDTW(1)] = main(test_data_temp,1);%1 stand for using tDTW
%     time_scale_tDTW(2,cnt) = toc;
%     cnt = cnt+1;
% end
% disp('data_scale2 has done!');
% 


load('data\Earthquakes.mat');
cnt = 1;
for m = 1:50
    test_data_temp = Earthquakes(1:m);
    tic
    [~,RI_aDTW(1)] = main(test_data_temp,0);%0 stand for using tDTW
    time_scale_aDTW(3,cnt) = toc;
    tic
    [~,RI_tDTW(1)] = main(test_data_temp,1);%1 stand for using tDTW
    time_scale_tDTW(3,cnt) = toc;
    cnt = cnt+1;
end
disp('data_scale3 has done!');

save data_scale
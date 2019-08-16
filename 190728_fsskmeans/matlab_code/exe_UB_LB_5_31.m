clear;clc;
RI_UB = zeros(1,15);
RI_LB = zeros(1,15);
RI_tDTW = zeros(1,15);
RI_aDTW = zeros(1,15);
RI_tDTW_auxi = zeros(1,15);
time_UB = zeros(1,15);
time_LB = zeros(1,15);
time_tDTW = zeros(1,15);
time_aDTW = zeros(1,15);
time_tDTW_auxi = zeros(1,15);

dataName = {'yoga'};


% dataName = {'Adiac','Earthquakes','ECG5000','FaceAll','MALLAT','MedicalImages','POC','Strawberry','wafer','yoga'};

for i = 1:length(dataName)
    load(['data\',char(dataName(i)),'.mat'])
%     tic
%     [Index_UB,RI_UB(i)] = main(eval(char(dataName(i))),2);%2 in main stand for upper bound
%     time_UB(i) = toc;

%     tic
%     [Index_LB,RI_LB(i)] = main(eval(char(dataName(i))),3);%3 in main stand for lower bound
%     time_LB(i) = toc;

%     tic
%     [Index_aDTW,RI_aDTW(i)] = main(eval(char(dataName(i))),0);
%     time_aDTW(i) = toc;

    tic
    [Index_aDTW,RI_aDTW(i),Index_aDTW_auxi,RI_aDTW_auxi(i)] = main(eval(char(dataName(i))),0);
    time_tDTW(i) = toc;

%     tic
%     [Index_tDTW_auxi,RI_tDTW_auxi(i)] = main(eval(char(dataName(i))),1.1);
%     time_tDTW_auxi(i) = toc;
    disp([char(dataName(i)),' has done!']); 
end
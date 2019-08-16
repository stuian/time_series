% using update V.S. not using update
RI_aDTW_kmeans_update = zeros(1,15);
RI_aDTW_kmeans_NoUpdate = zeros(1,15);
time_aDTW_kmeans_update = zeros(1,15);
time_aDTW_kmeans_NoUpdate = zeros(1,15);

load('data\MALLAT.mat');
testData = MALLAT;
tic
[~,RI_aDTW_kmeans_update(2)] = main(testData,0);%0 stand for aDTW
time_aDTW_kmeans_update(2) = toc;
tic
[~,RI_aDTW_kmeans_NoUpdate(2)] = main(testData,0.1);%0 stand for aDTW
time_aDTW_kmeans_NoUpdate(2) = toc;
disp('dataset2 has done!');


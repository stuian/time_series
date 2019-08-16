load('data\test_data.mat');
tic
[~,RandIndex] = main(test_data,0);%0 stand for aDTW
toc
plot(RandIndex,'x-');


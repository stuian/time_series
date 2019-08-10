function create = createDataSets()
indexes=[];
dataSetTrain=[];
dataSetTest=[];
totalElements=0;
classTrainSize=2;
classTestSize=5;
xz=load('xz.txt');
[MM idx]=unique(xz(:,1));
counts=diff([0;idx]);

for i=1:length(counts)
indexes=randperm(counts(i),classTrainSize+classTestSize);    
dataSetTrain= [dataSetTrain ; xz(indexes(1:classTrainSize)+totalElements,:)];
dataSetTest= [dataSetTest ; xz(indexes(classTrainSize+1:classTestSize+classTrainSize)+totalElements,:)];
totalElements=totalElements+counts(i);
end
dlmwrite('trainA.txt',dataSetTrain(:,1:end/2),'delimiter',' ');
dlmwrite('trainB.txt',dataSetTrain(:,(end/2)+1:end),'delimiter',' ');
dlmwrite('testA.txt',dataSetTest(:,1:end/2),'delimiter',' ');
dlmwrite('testB.txt',dataSetTest(:,(end/2)+1:end),'delimiter',' ');
end
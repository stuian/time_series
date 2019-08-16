% exe purity
dataName = {'Adiac','Earthquakes','ECG5000','FaceAll','MALLAT','MedicalImages','POC','Strawberry','wafer','yoga'};
purity = -ones(1,10);
nmi1 = -ones(1,10);
ARI = -ones(1,10);
for i = 1:length(dataName)
    load(['data\',char(dataName(i)),'.mat'])
    dataset = eval(char(dataName(i)));% get dataset
    labels = zeros(1,length(dataset));
    for k1=1:size(dataset,2)
        labels(1,k1)=dataset{k1}.label;
    end
        load (['Index_aDTW_',char(dataName(i))]) % get train_labels
        purity(i) = Purity(index, labels);
        % nmi = Nmi(Index_aDTW,labels);
        nmi1(i) = MutualInfo(index, labels, 'NMI');
        ARI(i) = adjrand(index, labels);
end







% RNNRatio
dataName = {'Adiac','Earthquakes','ECG5000','FaceAll','MALLAT','MedicalImages','POC','Strawberry','wafer','yoga'};
ratios = -ones(length(dataName),6);
for i = 1:length(dataName)
    load(['data\',char(dataName(i)),'.mat']);
    dataset = eval(char(dataName(i)));
    len = length(dataset);
    disMatrix = Dis_Matrix(dataset,2);% use ED
    nn = Nearest_neighbor(disMatrix);% 最近邻
    rnns = zeros(1,len); % 初始化反最近邻矩阵
    % 对每个数据点，找到它们的反最近邻
    for j = 1:len
        rnns(j) = length(Revers_nearest_neighbor(j,nn));
    end
    numOfRnn = unique(rnns);
    for k = 1:length(numOfRnn)
        ratios(i,numOfRnn(k)+1) = length(find(rnns == numOfRnn(k)))/len;
    end
    disp([char(dataName(i)), ' has finished!!!'])
end
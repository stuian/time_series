dataName = {'ECG5000'};
load(['data\',char(dataName(1)),'.mat'])
data_label = ECG5000;
data = zeros(length(data_label), length(data_label{1}.data));
for i = 1:length(data_label)
    data(i,:) = data_label{i}.data;
end
[pc,score,latent] = pca(data);
pc_choose = score(:,1:3);

% 
index = zeros(size(data_label));
for i = 1:length(data_label)
   index(i) = data_label{i}.label;
end

% restruct
findr = find(pc_choose(:,1)>-12 & pc_choose(:,1)<-4);
findb = find(pc_choose(:,1)>2 & pc_choose(:,1)<8);
findl = find(pc_choose(:,1)>-5 & pc_choose(:,1)<0 & pc_choose(:,2)>-10 & pc_choose(:,2)<-5);
% findl = find(pc_choose(:,1)<5);
% findm = find(pc_choose(:,1)<2 & pc_choose(:,1)>0 & pc_choose(:,2)<-4 & pc_choose(:,2)>-6);
% randNumr=randperm(length(findr));
% randNuml=randperm(length(findl));
% randNumm=randperm(length(findm));
% % 
% indexr=randNumr(1:20);
% indexl=randNuml(1:30);
% indexm=randNumm(1:2);
% % 
changer=findr;
changeb=findb;
changel = findl;
% changel=findl(indexl);
% changem=findm(indexm);

% for i = 1:length(changel)
%     Index_tDTW(changel(i)) = 1;
% end
for i = 1:length(changer)
    index(changer(i)) = 2;
end
for i = 1:length(changeb)
    index(changeb(i)) = 1;
end
for i = 1:length(changel)
    index(changel(i)) = 3;
end
% for i = 1:length(changem)
%     Index_tDTW(changem(i)) = 2;
% end

% index = Index_tDTW;
classes = unique(index);
num_classes = length(classes);

% 根据类别，分xyz并且绘制
x = cell(1,num_classes);
y = cell(1,num_classes);
z = cell(1,num_classes);
for i = 1:length(index)
    for j = 1:num_classes
        if index(i) == classes(j)
            x{j} = [x{j};pc_choose(i,1)];
            y{j} = [y{j};pc_choose(i,2)];
            z{j} = [z{j};pc_choose(i,3)];
        end
    end
end

for i = 1:length(x)
    temp = i;
    scatter3(x{temp},y{temp},z{temp})
    grid off
    hold on
end


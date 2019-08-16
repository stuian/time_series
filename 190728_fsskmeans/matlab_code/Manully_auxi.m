% Find some auxiliary information manually
% function [MustLink, CannotLink] = Manully_auxi(data)
data = eval(char(dataName(1)));
mustLink = {};
cannotLink = {};
MustLink = {};
CannotLink={};
h = 100;
for i = 1:length(data)-1
    for j = i+1:length(data)
        if Index_tDTW(i) ~= Index_tDTW(j) && data{i}.label == data{j}.label
            mustLink = [mustLink, [i,j]];
        elseif Index_tDTW(i) == Index_tDTW(j) && data{i}.label ~= data{j}.label
            cannotLink = [cannotLink, [i,j]];
        end
    end
end
% 从符合条件的数据对中随机选取h个mustlink和h个cannotlink
for i = 1:h
    randNum1 = int32(rand(1,1) * length(mustLink));
    MustLink = [MustLink,[mustLink{randNum1}]];
    randNum2 = int32(rand(1,1) * length(cannotLink));
    CannotLink = [CannotLink,[cannotLink{randNum2}]];
end


function purity = Purity(train_labels, labels)
    len = length(labels);% 数据点的数量
    classIndex = unique(train_labels);
    classNum = length(classIndex);% 算法得到的类的个数
    maxSum = 0;
    for i = 1:classNum % 对每个类进行处理和计算
        currIndex = classIndex(i); % 当前的类标号
        currDataIndex = train_labels == currIndex; % 当前算法分类标号的数据下标
        currTrueLabels = labels(currDataIndex); % 对应的真实分类标签
        currMax = length(find(currTrueLabels == mode(currTrueLabels)));
        maxSum = maxSum + currMax;
    end
    purity = maxSum/len;
end











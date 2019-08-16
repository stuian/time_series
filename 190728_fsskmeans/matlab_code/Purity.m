function purity = Purity(train_labels, labels)
    len = length(labels);% ���ݵ������
    classIndex = unique(train_labels);
    classNum = length(classIndex);% �㷨�õ�����ĸ���
    maxSum = 0;
    for i = 1:classNum % ��ÿ������д���ͼ���
        currIndex = classIndex(i); % ��ǰ������
        currDataIndex = train_labels == currIndex; % ��ǰ�㷨�����ŵ������±�
        currTrueLabels = labels(currDataIndex); % ��Ӧ����ʵ�����ǩ
        currMax = length(find(currTrueLabels == mode(currTrueLabels)));
        maxSum = maxSum + currMax;
    end
    purity = maxSum/len;
end











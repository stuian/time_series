% find revers nearest neighbor
function rnn = Revers_nearest_neighbor(query,NN)
% input: NNΪ�������queryΪ����Ѱ��rnn��data point
% output: ����query�ķ�����ڡ�
rnn = find(NN == query);
end
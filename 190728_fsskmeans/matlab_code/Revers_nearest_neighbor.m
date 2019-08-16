% find revers nearest neighbor
function rnn = Revers_nearest_neighbor(query,NN)
% input: NN为距离矩阵，query为请求寻找rnn的data point
% output: 返回query的反最近邻。
rnn = find(NN == query);
end
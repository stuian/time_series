% Auto auxiliary information
function [autoMust, autoCannot] = Auto_auxi(MustLink, CannotLink, nn)
% input: 人工加入的MustLink和CannotLink条件（都是1 x X的cell）
%        nn是data之间的距离矩阵。
% output: auto_auxi是自动生成的辅助条件。
x1 = size(MustLink, 2);% MustLink条件的个数
x2 = size(CannotLink, 2);% CannotLink条件的个数
% queries_Must = zeros(1, 2*x1);% 出现在人工辅助条件中的点的index
% queries_Cannot = zeros(1, 2*x2);
autoMust = {};% 自动的辅助条件
autoCannot = {};
% for i = 1:x1
%     queries_Must(2*i-1) = MustLink{i}(1);
%     queries_Must(2*i) = MustLink{i}(2);
% end
% for i = 1:x2
%     queries_Cannot(2*i-1) = CannotLink{i}(1);
%     queries_Cannot(2*i) = CannotLink{i}(2);
% end
for i = 1:x1
    rnn1 = Revers_nearest_neighbor(MustLink{i}(1),nn);
    rnn2 = Revers_nearest_neighbor(MustLink{i}(2),nn);
    if ~isempty(rnn1)
        for j = 1:length(rnn1)
            autoMust = [autoMust,[rnn1(j),MustLink{i}(1)]];
        end
    end
    if ~isempty(rnn2)
        for j = 1:length(rnn2)
            autoMust = [autoMust,[rnn2(j),MustLink{i}(2)]];
        end
    end
end

for i = 1:x2
    rnn1 = Revers_nearest_neighbor(CannotLink{i}(1),nn);
    rnn2 = Revers_nearest_neighbor(CannotLink{i}(2),nn);
    if ~isempty(rnn1)
        for j = 1:length(rnn1)
            autoCannot = [autoCannot,[rnn1(j),CannotLink{i}(2)]];
        end
    end
    if ~isempty(rnn2)
        for j = 1:length(rnn2)
            autoCannot = [autoCannot,[rnn2(j),CannotLink{i}(1)]];
        end
    end
end

end
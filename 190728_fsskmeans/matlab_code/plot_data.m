function plot_data(datacell)
n = length(datacell);
for i = 1:n
    if datacell{i}.label == 0
        plot(datacell{i}.data);
    end
    hold on;
end




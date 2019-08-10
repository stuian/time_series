function success = plotSuccess()
for i=40:40
    figure('units','normalized','outerposition',[0 0 0.7 0.7]);
iSuccess=load(strcat('iSuccess',num2str(i),'_1.txt'));
dSuccess=load(strcat('dSuccess',num2str(i),'_1.txt'));
alpha(0.4);
h=histfit(dSuccess(:,4)./dSuccess(:,7));

hold on;
h1=histfit(iSuccess(:,4)./iSuccess(:,7));

%h = findobj(gca,'Type','patch');
%h = findobj(gca,'Type','patch');
set(h(1),'FaceColor',[0 0 1],'EdgeColor',[0 0 1]);
set(h(2),'color',[0.5 0.3 0.5]);
alpha(0.5);
set(h1(1),'FaceColor',[1 0 0],'EdgeColor',[1 0 0]);
set(h1(2),'color',[1 0.3 0]);

threshold=load(strcat('threshold',num2str(i),'_1.txt'));
plot([threshold threshold],[0 20],'--','linewidth',3,'color','black');
saveppt('handWriting.ppt');

end
end
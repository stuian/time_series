function plotDI = plotWarp()
diff=[];
trainA=load('trainA.txt');
modelX=trainA(1,2:end);
modelY=trainA(2,2:end);
for i=3:2:size(trainA,1)-1
   iScore=dtw((trainA(i,2:end))',modelX',50)+dtw((trainA(i+1,2:end))',modelY',50);
   dScore=dtw([(trainA(i,2:end));(trainA(i+1,2:end))]',[modelX;modelY]',50);
   diff(floor(i/2.00))=abs(dScore-iScore);
end

plot(diff,'linewidth',3);

end
function new = shiftData()
xright=load('zright.txt');
shift=ceil(rand(1,1)*300);
temp=[];
for i=1:size(xright,1)
    temp=xright(i,end-shift+1:end);
    xright(i,shift+2:end)=xright(i,2:end-shift);
    xright(i,2:shift+1)=temp;
end
dlmwrite('zrightShifted.txt',xright,'delimiter',' ');
end
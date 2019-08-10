function shuffle = shuffleData()

n1=ceil(rand(1,4000)*2280);
n2=ceil(rand(1,4000)*2280);
xtrain=load('z_right.txt');
x=xtrain;
% xtest=load('x_test.txt');
% x=[xtrain;xtest];
ytrain=load('z_left.txt');
y=ytrain;
% ytest=load('y_test.txt');
% y=[ytrain;ytest];
% ztrain=load('z_train.txt');
% ztest=load('z_test.txt');
% z=[ztrain;ztest];

for i=1:length(n1)
   temp=x(n1(i),:);
   x(n1(i),:)=x(n2(i),:);
   x(n2(i),:)=temp;
   
   temp=y(n1(i),:);
   y(n1(i),:)=y(n2(i),:);
   y(n2(i),:)=temp;
   
%    temp=z(n1(i),:);
%    z(n1(i),:)=z(n2(i),:);
%    z(n2(i),:)=temp;
end

dlmwrite('zrighttrain.txt',x(1:300,:),'delimiter',' ');
dlmwrite('zrighttest.txt',x(301:800,:),'delimiter',' ');
dlmwrite('zlefttrain.txt',y(1:300,:),'delimiter',' ');
dlmwrite('zlefttest.txt',y(301:800,:),'delimiter',' ');
% dlmwrite('ztrain.txt',z(1:150,:),'delimiter',' ');
% dlmwrite('ztest.txt',z(151:1000,:),'delimiter',' ');
end
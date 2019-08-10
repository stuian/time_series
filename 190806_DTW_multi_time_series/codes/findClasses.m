function check = findClasses()

testClassesD=load('testClassesD.txt');
testClassesI=load('testClassesI.txt');
distanceInfo=load('distanceInfo.txt');
m=[];
c=0;
for i=1:size(testClassesD,1)
   if (testClassesI(i,1)~=testClassesI(i,2) && testClassesD(i,1)==testClassesD(i,2))
       c=c+1;
       m(c,:)=[testClassesI(i,1) testClassesI(i,2) testClassesD(i,2) distanceInfo(i,:)];
   end   
end
dlmwrite('dSuccess.txt',m,'delimiter',' ');

m=[];
c=0;
for i=1:size(testClassesI,1)
   if (testClassesI(i,1)==testClassesI(i,2) && testClassesD(i,1)~=testClassesD(i,2))
       c=c+1;
       m(c,:)=[testClassesI(i,1) testClassesI(i,2) testClassesD(i,2) distanceInfo(i,:)];
   end   
end
dlmwrite('iSuccess.txt',m,'delimiter',' ');
end
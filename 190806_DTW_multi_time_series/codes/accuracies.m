function [X Y I D Best myAlgorithm successI successD successAlgorithm]=accuracies()
threshold=find_threshold();
dlmwrite('threshold.txt',threshold);
testClassesD=load('testClassesD.txt');
testClassesI=load('testClassesI.txt');
testClassesX=load('testClassesX.txt');
testClassesY=load('testClassesY.txt');
distanceInfo=load('distanceInfo.txt');
truePositivesI=0;
truePositivesD=0;
truePositivesX=0;
truePositivesY=0;
truePositivesBest=0;
truePositivesmyAlgorithm=0;
myAlgorithmSuccess=0;

for i=1:size(testClassesI,1)
    if (testClassesD(i,1)==testClassesD(i,2))
        truePositivesD=truePositivesD+1;
    end
    if (testClassesI(i,1)==testClassesI(i,2))
        truePositivesI=truePositivesI+1;
    end
    if (testClassesX(i,1)==testClassesX(i,2))
        truePositivesX=truePositivesX+1;
    end
    if (testClassesY(i,1)==testClassesY(i,2))
        truePositivesY=truePositivesY+1;
    end
    if (testClassesI(i,1)==testClassesI(i,2) || testClassesD(i,1)==testClassesD(i,2))
        truePositivesBest=truePositivesBest+1;
    end
    if ((((distanceInfo(i,1)/distanceInfo(i,4))<=threshold) && (testClassesD(i,1)==testClassesD(i,2))) ||...
            (((distanceInfo(i,1)/distanceInfo(i,4))>threshold) && (testClassesI(i,1)==testClassesI(i,2))))
        truePositivesmyAlgorithm=truePositivesmyAlgorithm+1;
    end
    if ((((distanceInfo(i,1)/distanceInfo(i,4))<=threshold) && (testClassesD(i,1)==testClassesD(i,2)) && (testClassesI(i,1)~=testClassesI(i,2))) ||...
            (((distanceInfo(i,1)/distanceInfo(i,4))>threshold) && (testClassesI(i,1)==testClassesI(i,2)) && (testClassesD(i,1)~=testClassesD(i,2))))
        myAlgorithmSuccess=myAlgorithmSuccess+1;
    end
end
I = double(truePositivesI)/size(testClassesI,1);
D = double(truePositivesD)/size(testClassesI,1);
X = double(truePositivesX)/size(testClassesI,1);
Y = double(truePositivesY)/size(testClassesI,1);
Best = double(truePositivesBest)/size(testClassesI,1);
myAlgorithm = double(truePositivesmyAlgorithm)/size(testClassesI,1);

iSuccess=load('iSuccess.txt');
dSuccess=load('dSuccess.txt');
successI=size(iSuccess,1);
successD=size(dSuccess,1);
successAlgorithm=myAlgorithmSuccess;
end
function [accuracyI accuracyD]=classificationTest()

w=30; % warping window constraint
train1=load('trainA.txt');
train2=load('trainB.txt');
test1=load('testA.txt');
test2=load('testB.txt');



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
classesI(95,2)=0;
classesD(95,2)=0;
% confusionI(95,95)=0; %confusion matrix
% confusionD(95,95)=0; %confusion matrix



testClassesI(size(test1,1),2)=0;
testClassesD(size(test1,1),2)=0;
testClassesX(size(test1,1),2)=0;
testClassesY(size(test1,1),2)=0;
distanceInfo(size(test1,1),16)=0;
%%%% go through the whole test set
truePositivesI=0;
truePositivesD=0;
truePositivesBest=0;
for i=1:size(test1,1)
    i
    
    minDistanceI=100000;
    minDistanceD=100000;
    minDistanceX=100000;
    minDistanceY=100000;
    classI=0;
    classD=0;
    classX=0;
    classY=0;
    dist1=[];
    dist2=[];
    distD=[];
    ij=0;
    dj=0;
    fit1=[];
    fit2=[];
    %%% iterate through the whole train set
    for j=1:size(train1,1)
        if (i~=j)
        [distanceX dist1 path1]=dtw((test1(i,2:end))',(train1(j,2:end))',w);
        [distanceY dist2 path2]=dtw((test2(i,2:end))',(train2(j,2:end))',w);
        distanceI = distanceX+distanceY;
        [distanceD distD pathD] = dtw([test1(i,2:end);test2(i,2:end)]',[train1(j,2:end);train2(j,2:end)]',w);
        
        if (distanceI<minDistanceI)
            minDistanceI=distanceI;
            classI = train1(j,1);
            ij=j;
            %[d d1 d11 d22]=dtwI((test1(i,2:end))',(train1(j,2:end))',(test2(i,2:end))',(train2(j,2:end))',15);
        end
        if (distanceD<minDistanceD)
            minDistanceD=distanceD;
            classD = train1(j,1);
            dj=j;
        end
        if (distanceX<minDistanceX)
            minDistanceX=distanceX;
            classX=train1(j,1);
        end
        if (distanceY<minDistanceY)
            minDistanceY=distanceY;
            classY=train1(j,1);
        end
        end
    end
    distanceInfo(i,1)=minDistanceD;
    distanceInfo(i,2)=0;
    distanceInfo(i,3)=0;
    distanceInfo(i,4)=minDistanceI;
    distanceInfo(i,5)=0;
    distanceInfo(i,6)=0;
    distanceInfo(i,7)=abs(minDistanceD/minDistanceI);
    distanceInfo(i,8)=sqrt(sum(diff(test1(i,2:end)).^2));
    distanceInfo(i,9)=sqrt(sum(diff(test2(i,2:end)).^2));
    %%%% calculating the bit costs for the paths %%%%%%%%%
    %     dist1=zscore(dist1);
    %     dist2=zscore(dist2);
    %distD=zscore(distD);
    %      dist1=DNorm_Unif(dist1);
    %      dist2=DNorm_Unif(dist2);
    %      distD=DNorm_Unif(distD);
    %      distanceInfo(i,10)=calculate_model_cost(dist1)+calculate_model_cost(dist2);
    %      distanceInfo(i,11)=calculate_model_cost(distD);
    %     distanceInfo(i,12)=length(dist1);
    %     distanceInfo(i,13)=length(dist2);
    %     distanceInfo(i,14)=length(distD);
    %distanceInfo(i,10)=dtw(path1',path2',30);
    %fit1 = dtwFit((test1(i,2:end))',(train1(dj,2:end))',pathD);
    %fit2 = dtwFit((test2(i,2:end))',(train2(dj,2:end))',pathD);
    %distanceInfo(i,10)=sum(abs(fit1-fit2));
    % fit1=dtwFit([test1(i,2:end);test2(i,2:end)]',[train1(dj,2:end);train2(dj,2:end)]',path1);
    % fit2=dtwFit([test1(i,2:end);test2(i,2:end)]',[train1(dj,2:end);train2(dj,2:end)]',path2);
    %        fit1=DNorm_Unif(fit1);
    %        fit2=DNorm_Unif(fit2);
    %fit12=abs(fit1-fit2);
    %distanceInfo(i,10)=calculate_model_cost(fit12);
    %       distanceInfo(i,11)=calculate_model_cost(fit2);
    %      fitBits=calculate_model_cost(fit1)+calculate_model_cost(fit2);
    % fitBits=sum(fit1)+sum(fit2);
    %     distanceInfo(i,10)=fitBits;
    %     fit1=DNorm_Unif(fit1);
    %     fit2=DNorm_Unif(fit2);
    %     %modelCost = calculate_model_cost(dist1)+calculate_model_cost(dist2);
    %     mdlCost = calculate_model_cost(fit1)+calculate_model_cost(fit2);
    %     distanceInfo(i,16)= mdlCost;
    
    if (test1(i,1)==classI)
        truePositivesI=truePositivesI+1;
        classesI(test1(i,1),1)=classesI(test1(i,1),1)+1;
    end
    if (test1(i,1)==classD)
        truePositivesD=truePositivesD+1;
        classesD(test1(i,1),1)=classesD(test1(i,1),1)+1;
    end
    if(test1(i,1)==classI || test1(i,1)==classD)
        truePositivesBest=truePositivesBest+1;
    end
    testClassesI(i,1)=test1(i,1);
    testClassesI(i,2)=classI;
    testClassesD(i,1)=test1(i,1);
    testClassesD(i,2)=classD;
    testClassesX(i,1)=test1(i,1);
    testClassesX(i,2)=classX;
    testClassesY(i,1)=test1(i,1);
    testClassesY(i,2)=classY;
    %     confusionI(test1(i,1),classI) = confusionI(test1(i,1),classI)+1;
    %     confusionD(test1(i,1),classD) = confusionD(test1(i,1),classD)+1;
    classesI(test1(i,1),2)=classesI(test1(i,1),2)+1;
    classesD(test1(i,1),2)=classesD(test1(i,1),2)+1;
    %dlmwrite('classesI.txt',classesI,'delimiter',' ');
    %dlmwrite('classesD.txt',classesD,'delimiter',' ');
    %dlmwrite('confusionMatrixI.txt',confusionI,'delimiter',' ');
    %dlmwrite('confusionMatrixD.txt',confusionD,'delimiter',' ');
    dlmwrite('testClassesI.txt',testClassesI,'delimiter',' ');
    dlmwrite('testClassesD.txt',testClassesD,'delimiter',' ');
    dlmwrite('testClassesX.txt',testClassesX,'delimiter',' ');
    dlmwrite('testClassesY.txt',testClassesY,'delimiter',' ');
    dlmwrite('distanceInfo.txt',distanceInfo,'delimiter',' ');
end
accuracyI=double(truePositivesI)/size(test1,1);
accuracyD=double(truePositivesD)/size(test1,1);
accuracyBest=double(truePositivesBest)/size(test1,1);
findClasses();
[X Y I D Best myAlgorithm successI successD successAlgorithm]=accuracies();
dlmwrite('Results.txt',X);
dlmwrite('Results.txt',Y,'-append');
dlmwrite('Results.txt',I,'-append');
dlmwrite('Results.txt',D,'-append');
dlmwrite('Results.txt',Best,'-append');
dlmwrite('Results.txt',myAlgorithm,'-append');
dlmwrite('Results.txt',successI,'-append');
dlmwrite('Results.txt',successD,'-append');
dlmwrite('Results.txt',successAlgorithm,'-append');
end
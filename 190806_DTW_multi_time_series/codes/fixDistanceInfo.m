function fix=fixDistanceInfo()

distanceInfo=load('distanceInfo.txt');
distanceInfo(:,5)=distanceInfo(:,4);
distanceInfo(:,6)=distanceInfo(:,4);
distanceInfo(:,7)=abs(distanceInfo(:,1)-distanceInfo(:,4));
% distanceInfo(:,7)=distanceInfo(:,7)./distanceInfo(:,4);
dlmwrite('distanceInfo.txt',distanceInfo,'delimiter',' ');

end
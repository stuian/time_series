clear all
clc
close all
mdist=input('输入文件名\n');
disp('read')

xx=load(mdist);

[number, row]=size(xx);

yy=pdist(xx,'euclidean');
%获取聚类，第二参数指定层次聚类方式
%'single'：单连通，最短距离法（默认）；'complete'：全连通，最长距离法；'average'：未加权平均距离法； 
%'weighted'： 加权平均法；'centroid'： 质心距离法；'median'：加权质心距离法；'ward'：内平方距离法（最小方差算法）
zz=linkage(yy,'single');
%指定获取簇类个数
Ncluster=input('输入类个数\n');
%获取指定Ncluster个数的聚类结果
c = cluster( zz,'maxclust', Ncluster ); 



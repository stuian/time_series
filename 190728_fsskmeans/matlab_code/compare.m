clear all
clc
close all
mdist=input('�����ļ���\n');
disp('read')

xx=load(mdist);

[number, row]=size(xx);

yy=pdist(xx,'euclidean');
%��ȡ���࣬�ڶ�����ָ����ξ��෽ʽ
%'single'������ͨ����̾��뷨��Ĭ�ϣ���'complete'��ȫ��ͨ������뷨��'average'��δ��Ȩƽ�����뷨�� 
%'weighted'�� ��Ȩƽ������'centroid'�� ���ľ��뷨��'median'����Ȩ���ľ��뷨��'ward'����ƽ�����뷨����С�����㷨��
zz=linkage(yy,'single');
%ָ����ȡ�������
Ncluster=input('���������\n');
%��ȡָ��Ncluster�����ľ�����
c = cluster( zz,'maxclust', Ncluster ); 



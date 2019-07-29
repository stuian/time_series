"""
date:20190718
author:Jade
theme:基于密度的聚类算法
"""
#输出标记数组，每个对象属于每个簇或者为噪声点

import numpy as np
import matplotlib.pyplot as plt
import math
import random
from sklearn import datasets

class visitlist(object):
    def __init__(self,count=0):
        self.unvisitedlist = [i for i in range(count)]#记录未访问过的点
        self.visitedlist = list()#记录已经访问过的点
        self.unvisitednum = count #记录未访问过的点数量

    def visit(self,pointId):
        self.visitedlist.append(pointId)
        self.unvisitedlist.remove(pointId)
        self.unvisitednum -= 1

def dist(a,b):
    """
    计算两个元组之间的欧几里得距离
    """
    return np.sqrt(np.power(a-b,2).sum())

def dbscan(dataSet,eps,minPts):
    """
    :param dataSet:
    :param eps: 0.1
    :param minPts: 10
    :return:
    """
    nPoints = dataSet.shape[0]
    vPoints = visitlist(count=nPoints)
    k = -1
    #初始所有数据标记为-1
    C = [-1 for i in range(nPoints)]
    while vPoints.unvisitednum>0:
        P = random.choice(vPoints.unvisitedlist)
        vPoints.visit(P)
        # N：求P的邻域
        N = [i for i in range(nPoints) if dist(dataSet[i],dataSet[P])<=eps]
        if len(N) >= minPts: #P的邻域里至少有minPts个对象
            # 创建新簇，把P添加到新簇
            k += 1
            C[P] = k
            for pl in N:
                if pl in vPoints.unvisitedlist:
                    vPoints.visit(pl)
                    M = [i for i in range(nPoints) if dist(dataSet[i],dataSet[pl]) <= eps]
                    if len(M) >= minPts:
                        for i in M:
                            if i not in N:
                                N.append(i) # N长度增加，循环次数也增多了
                    if C[pl] == -1:
                        C[pl] = k
        else:
            C[P] = -1
    return C

if __name__ == '__main__':
    X1, Y1 = datasets.make_circles(n_samples=2000, factor=0.6, noise=0.05,random_state=1)
    X2, Y2 = datasets.make_blobs(n_samples=500, n_features=2, centers=[[1.5,1.5]],cluster_std=[[0.1]], random_state=5)

    X = np.concatenate((X1,X2))
    y_pred = dbscan(X,0.1,10)
    plt.figure(figsize=(12,9),dpi=80)
    plt.scatter(X[:,0], X[:,1],c=y_pred,marker='.')
    plt.show()
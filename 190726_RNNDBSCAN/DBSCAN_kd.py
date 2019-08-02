import numpy as np
import matplotlib.pyplot as plt
import math
import random
from scipy.spatial import KDTree
import time
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

def dbscan2(dataSets,eps,minPts):
    nPoints = dataSets.shape[0]
    vPoints = visitlist(count=nPoints)
    k = -1
    C = [-1 for i in range(nPoints)]
    kd = KDTree(X) #二叉搜索树在多维的推广
    while (vPoints.unvisitednum > 0):
        p = random.choice(vPoints.unvisitedlist)
        vPoints.visit(p)
        N = kd.query_ball_point(dataSets[p], eps)
        if len(N) >= minPts:
            k += 1
            C[p] = k
            for p1 in N:
                if p1 in vPoints.unvisitedlist:
                    vPoints.visit(p1)
                    M = kd.query_ball_point(dataSets[p1], eps)
                    if len(M) >= minPts:
                        for i in M:
                            if i not in N:
                                N.append(i)
                    if C[p1] == -1:
                        C[p1] = k
        else:
            C[p1] = -1
    return C

X1, Y1 = datasets.make_circles(n_samples=2000, factor=0.6, noise=0.05,random_state=1)
X2, Y2 = datasets.make_blobs(n_samples=500, n_features=2, centers=[[1.5,1.5]],cluster_std=[[0.1]], random_state=5)
X = np.concatenate((X1,X2))

start = time.time()
y_pred = dbscan2(X,0.1,10)
end = time.time()
print("运行时间：",end-start)

plt.figure(figsize=(12,9),dpi=80)
plt.scatter(X[:,0], X[:,1],c=y_pred,marker='.')
plt.show()
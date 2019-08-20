import numpy as np
import time
from evaluation import Purity
from evaluation import RandIndex
from evaluation import NMI
import h5py
import os

def init_random_medoids(X,k):
    centroids = np.zeros((k, X.shape[1], X.shape[2]))
    for i in range(k):
        centroid = X[np.random.choice(range(X.shape[0]))]
        centroids[i] = centroid
    return centroids

def dist(a,b):
    return np.sqrt(sum(np.power(a - b, 2)))

def DTWd(X,Y):
    m = X.shape[0]
    n = Y.shape[0]
    matrix_DTW = {}
    for i in range(m):
        matrix_DTW[(i, -1)] = float('inf')
    for i in range(n):
        matrix_DTW[(-1, i)] = float('inf')
    matrix_DTW[(-1, -1)] = 0
    for i in range(m):
        for j in range(n):
            matrix_DTW[(i, j)] = dist(X[i, :], Y[j, :]) + min(matrix_DTW[(i - 1, j)], matrix_DTW[(i, j - 1)],
                                                              matrix_DTW[(i - 1, j - 1)])
    return np.sqrt(matrix_DTW[(m - 1, n - 1)])

def update_centroids(X,clusters,centroids):
    # update Center point
    for i in range(len(clusters)):
        temp = np.zeros(centroids[0].shape)
        for j in clusters[i]:
            temp = temp + X[j]
        temp = temp / len(clusters[i]) # 初始化随机取到极端点，会导致len(clusters)为0
        # print("类%d的样本个数：" % i,len(clusters[i]))
        centroids[i] = temp
    return centroids

def kmeans_DTWd(X,y,centroids):
    k = len(np.unique(y))
    m = X.shape[0]

    y_pred = np.zeros(m)
    assignment = [[] for _ in range(k)]
    # centroids = init_random_medoids(X,k)

    clusterChanged = True
    start = time.time()
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = np.inf
            minIndex = -1
            for j in range(k):
                distJI = DTWd(centroids[j],X[i])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if y_pred[i] != minIndex:
                clusterChanged = True  # 一直到收敛，不在更新，则迭代完成
            y_pred[i] = minIndex
            assignment[minIndex].append(i)
        # clusterChanged += 1
        centroids = update_centroids(X,assignment,centroids)
    end = time.time()
    return float(end-start),y_pred,RandIndex(y_pred,y),Purity(y_pred,y),NMI(y_pred,y)


def main():
    # 1、get Robot Execution Failures lp1-5 data
    data_name = ['Robot Execution Failures lp1', 'Robot Execution Failures lp2', 'Robot Execution Failures lp3','Robot Execution Failures lp4','Robot Execution Failures lp5']
    count = 6
    for file in data_name:
        filename = file + '.h5'
        f = h5py.File(filename,'r')
        X = f['train_x'][:]
        y = f['train_y'][:] # 1,2,3... np.array;y_pred:0,1,2
        cost_time,y_pred,randindex,purity,nmi= kmeans_DTWd(X,y)
        newfilename = str(count) + '.npy'
        path = './result'
        newpath = os.path.join(path, newfilename)
        count += 1
        np.save(newpath,y_pred)
        print(file,"数据集聚类的时间为%.2f秒,RI值为%.2f,purity值为%.2f,nmi值为%.2f" % (cost_time,randindex,purity,nmi))

    # 2、

if __name__ == '__main__':
    main()




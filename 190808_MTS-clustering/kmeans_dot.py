import numpy as np
import time
from evaluation import Purity
from evaluation import RandIndex
from evaluation import NMI
import h5py
import os
from shape_based_distance import multi_simlarity

def init_random_medoids(X,k):
    centroids = np.zeros((k, X.shape[1], X.shape[2]))
    for i in range(k):
        centroid = X[np.random.choice(range(X.shape[0]))]
        centroids[i] = centroid
    return centroids

def update_centroids(X,clusters,centroids):
    # update Center point
    for i in range(len(clusters)):
        temp = np.zeros(centroids[0].shape)
        for j in clusters[i]:
            temp = temp + X[j]
        temp = temp / len(clusters[i])
        centroids[i] = temp
    return centroids

def kmeans_dot(X,y,centroids):
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
                distJI = multi_simlarity(centroids[j],X[i],choose_min=False)
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
    count = 21
    for file in data_name:
        filename = file + '.h5'
        f = h5py.File(filename,'r')
        X = f['train_x'][:]
        y = f['train_y'][:] # 1,2,3... np.array
        cost_time,y_pred,randindex,purity,nmi= kmeans_dot(X,y) # centroids
        newfilename = str(count) + '.npy'
        path = './result'
        newpath = os.path.join(path,newfilename)
        count += 1
        np.save(newpath,y_pred)
        print(file,"数据集聚类的时间为%.2f秒,RI值为%.2f,purity值为%.2f,nmi值为%.2f" % (cost_time,randindex,purity,nmi))

    # 2、

if __name__ == '__main__':
    main()
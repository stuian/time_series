import numpy as np
import time
from evaluation import Purity
from evaluation import RandIndex
from evaluation import NMI
import h5py
import os
import heapq

def init_random_medoids(X,k):
    centroids = np.zeros((k, X.shape[1], X.shape[2]))
    for i in range(k):
        centroid = X[np.random.choice(range(X.shape[0]))]
        centroids[i] = centroid
    return centroids

def DTW(s1,s2):
    DTW = {}
    for i in range(len(s1)):
        DTW[(i, -1)] = float('inf')
    for i in range(len(s2)):
        DTW[(-1, i)] = float('inf')
    DTW[(-1, -1)] = 0
    for i in range(len(s1)):
        for j in range(len(s2)):
            dist = (s1[i] - s2[j]) ** 2
            DTW[(i, j)] = dist + min(DTW[(i - 1, j)], DTW[(i, j - 1)], DTW[(i - 1, j - 1)])
    return np.sqrt(DTW[(len(s1) - 1, len(s2) - 1)])

def DTWi(X,Y):
    distance = 0
    n = X.shape[1]
    for i in range(n):
        temp = DTW(X[:,i],Y[:,i])
        distance = distance + temp
    return distance

def dist(a,b):
    return np.sqrt(sum(np.power(a - b, 2)))

def lb_keogh(s1,s2):
    r = len(s1) // 7  # ？为什么除以7
    LB_sum = 0
    n = len(s2)
    for ind, i in enumerate(s1):
        lower_bound = min(s2[(ind - r if ind - r >= 0 else 0):(ind + r if ind + r <= n else n)])
        upper_bound = max(s2[(ind - r if ind - r >= 0 else 0):(ind + r if ind + r <= n else n)])
        if i > upper_bound:
            LB_sum += (i - upper_bound) ** 2
        elif i < lower_bound:
            LB_sum += (i - lower_bound) ** 2
    return np.sqrt(LB_sum)

def aDTWi(Q, C,theta):
    UB = 0
    LB = 0
    for i in range(Q.shape[1]):
        temp1 = dist(Q[:,i],C[:,i])
        temp2 = lb_keogh(Q[:,i],C[:,i])
        UB = UB + temp1
        LB = LB + temp2
    return LB + theta * (UB - LB)

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def update_centroids(X,clusters,centroids):
    # update Center point
    for i in range(len(clusters)):
        temp = np.zeros(centroids[0].shape)
        for j in clusters[i]:
            temp = temp + X[j]
        temp = temp / len(clusters[i])
        centroids[i] = temp
    return centroids

def kmeans_aDTWi(X,y,centroids):
    k = len(np.unique(y))
    m = X.shape[0]
    theta = 0.3

    y_pred = np.zeros(m)
    assignment = [[] for _ in range(k)]
    # centroids = init_random_medoids(X,k)

    # calculate distance matrix
    distance_matrix = np.zeros((k, m))
    for i in range(k):
        for j in range(m):
            distance_matrix[i, j] = aDTWi(centroids[i], X[j], theta)
    sigma = sigmoid(np.std(distance_matrix))

    clusterChanged = True
    start = time.time()
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            distances = distance_matrix[:, i]
            temp = list(map(list(distances).index, heapq.nsmallest(2, distances)))
            index1 = temp[0]
            min1 = distances[index1]
            index2 = temp[1]
            min2 = distances[index2]
            if abs(min2 - min1) >= sigma:
                minIndex = index1
            else:
                min1 = DTWi(centroids[index1], X[i])
                min2 = DTWi(centroids[index2], X[i])
                if min1 < min2:
                    minIndex = index1
                else:
                    minIndex = index2
            if y_pred[i] != minIndex:
                clusterChanged = True  # 一直到收敛，不在更新，则迭代完成
            y_pred[i] = minIndex
            assignment[minIndex].append(i)
            # clusterChanged += 1
        centroids = update_centroids(X, assignment, centroids)
        for i in range(k):
            for j in range(m):
                distance_matrix[i, j] = aDTWi(centroids[i], X[j], theta)
        sigma = sigmoid(np.std(distance_matrix))
    end = time.time()
    return float(end-start),y_pred,RandIndex(y_pred,y),Purity(y_pred,y),NMI(y_pred,y)


def main():
    # 1、get Robot Execution Failures lp1-5 data
    data_name = ['Robot Execution Failures lp1', 'Robot Execution Failures lp2', 'Robot Execution Failures lp3','Robot Execution Failures lp4','Robot Execution Failures lp5']
    count = 11
    for file in data_name:
        filename = file + '.h5'
        f = h5py.File(filename,'r')
        X = f['train_x'][:]
        y = f['train_y'][:] # 1,2,3... np.array
        cost_time,y_pred,randindex,purity,nmi= kmeans_aDTWi(X,y)
        newfilename = str(count) + '.npy'
        path = './result'
        newpath = os.path.join(path,newfilename)
        count += 1
        np.save(newpath,y_pred)
        print(file,"数据集聚类的时间为%.2f秒,RI值为%.2f,purity值为%.2f,nmi值为%.2f" % (cost_time,randindex,purity,nmi))

    # 2、

if __name__ == '__main__':
    main()




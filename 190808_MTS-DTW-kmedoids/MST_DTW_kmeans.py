"""
date:20190811
author:Jade
theme:多元时间序列的DTW和aDTW
"""

import numpy as np
import time
from evaluation import Purity
from evaluation import RandIndex
from evaluation import NMI
import h5py
import heapq
from multiprocessing import Process

class MST_kmeans(Process):
    def __init__(self, X,y,k=2, max_iterations=500,method=1,beta = None):
        super().__init__()
        self.X = X
        self.y = y
        self.k = k
        self.max_iterations = max_iterations
        self.method = method
        self.beta = beta

    def init_random_medoids(self):
        n_samples,length,n_features = np.shape(self.X)
        medoids = np.zeros((self.k,length,n_features))
        for i in range(self.k):
            medoid = self.X[np.random.choice(range(n_samples))]
            medoids[i] = medoid
        return medoids

    def DTW(self,s1,s2):
        """
        计算两个向量的DTW值
        :param s1: 向量1;list和array类型都可以
        :param s2: 向量2
        :return:
        """
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

    def lb_keogh(self,s1,s2):
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

    def dist(self,a,b):
        return np.sqrt(sum(np.power(a - b, 2)))

    def aDTW(self,s1, s2, beta):
        LB = self.lb_keogh(s1, s2)  # lower bound
        UB = self.dist(s1, s2)  # upper bound
        return LB + beta * (UB - LB)

    def similarity_distance(self,sample,medoids):
        distances = np.zeros(self.k)
        if self.method == 1:
            # DTWi
            for i in range(self.k):
                medoid = medoids[i]
                distance = 0
                n_features = sample.shape[1]
                for j in range(n_features):
                    temp = self.DTW(sample[:,j], medoid[:,j])
                    distance = distance + temp
                distances[i] = distance
        if self.method == 2:
            # DTWd
            for index in range(self.k):
                medoid = medoids[index]
                m = sample.shape[0]
                n = medoid.shape[0]
                matrix_DTW = {}
                for i in range(m):
                    matrix_DTW[(i, -1)] = float('inf')
                for i in range(n):
                    matrix_DTW[(-1, i)] = float('inf')
                matrix_DTW[(-1, -1)] = 0
                for i in range(m):
                    for j in range(n):
                        matrix_DTW[(i, j)] = self.dist(sample[i, :], medoid[j, :]) + min(matrix_DTW[(i - 1, j)], matrix_DTW[(i, j - 1)],matrix_DTW[(i - 1, j - 1)])
                distances[index] = np.sqrt(matrix_DTW[(m - 1, n - 1)])
        if self.method == 3:
            # aDTWi
            for i in range(self.k):
                medoid = medoids[i]
                distance = 0
                n_features = sample.shape[1]
                for j in range(n_features):
                    temp = self.aDTW(sample[:,j], medoid[:,j])
                    distance = distance + temp
                distances[i] = distance
        return distances

    def sigmoid(self,x):
        return 1.0 / (1 + np.exp(-x))

    def closest_medoid(self, sample, medoids):
        distances = self.similarity_distance(sample, medoids)
        if self.method < 3:
            closest_i = np.argmin(distances)
        else:
            sigma = self.sigmoid(np.std(distances))
            temp = list(map(list(distances).index, heapq.nsmallest(2, distances)))
            index1 = temp[0]
            min1 = distances[index1]
            index2 = temp[1]
            min2 = distances[index2]
            if abs(min2 - min1) >= sigma:
                closest_i = index1
            else:
                distance = 0
                d = []
                for i in [index1,index2]:
                    n_features = sample.shape[1]
                    for j in range(n_features):
                        temp = self.DTW(sample[:, j], medoids[i][:, j])
                        distance = distance + temp
                    d.append(distances)
                min1 = d[0]
                min2 = d[1]
                if min1 < min2:
                    closest_i = index1  # 分类标记
                else:
                    closest_i = index2
        return closest_i

    def create_clustersandindex(self,medoids):
        index = np.zeros(self.X.shape[0])
        clusters = [[] for _ in range(self.k)]
        for sample_i, sample in enumerate(self.X):
            medoid_i = self.closest_medoid(sample, medoids)
            clusters[medoid_i].append(sample_i)
            index[sample_i] = medoid_i
        return clusters,index

    def run(self):
        medoids = self.init_random_medoids(self.X)
        start = time.time()
        for iter in range(self.max_iterations):
            clusters,index = self.create_clustersandindex(self.X, medoids)

            # update Center point
            for k1 in range(self.k):
                clust_ind = []
                for ind,j in enumerate(index):
                    if j == k1:
                        clust_ind.append(ind)
                temp = np.zeros(medoids[0].shape)
                for k2 in clust_ind:
                    temp = temp + self.X[k2]
                temp = temp / len(clust_ind)
                medoids[k1] = temp  # 更新
            RI = RandIndex(index,self.y)
            purity = Purity(index,self.y)
            nmi = NMI(index,self.y)
            print("RI,purity,NMI of iter %d is %.2f,%.2f,%.2f respectively",(iter,RI,purity,nmi))
        end = time.time()
        print(end-start)
        file = h5py.File('cricket_predict.h5', 'w')
        file.create_dataset('pred_y', data=index)

def main():
    file = h5py.File('cricket_data.h5','r')
    X = file['train_x'][:]
    y = file['train_y'][:]
    file.close()
    k = len(np.unique(y))
    iters = 20
    p1 = MST_kmeans(X,y,k,iters,1)
    p1.start()
    p2 = MST_kmeans(X,y,k,iters,2)
    p2.start()
    p3 = MST_kmeans(X,y,k,iters,3,0.2)
    p3.start()
    p1.join()
    p2.join()
    p3.join()
    print('end!')


if __name__ == '__main__':
    main()
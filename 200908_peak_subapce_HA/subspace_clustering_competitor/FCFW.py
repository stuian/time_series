import numpy as np
import os
import h5py
from densityPeakRNN import distance_to_HDP
import pandas as pd
import random
import operator
import math
import time
from evaluation import RandIndex
from evaluation import Purity
from evaluation import NMI

def SDB(x,y):
    # 越小越相似
    y_ = [i for i in reversed(y)]
    CC = np.convolve(x,y_)
    s = np.linalg.norm(x) * np.linalg.norm(y)
    if s == 0 :
        value = 0
    else:
        NCC = CC / s
        index = np.argmax(NCC)
        value = NCC[index]
    dist = 1 - value
    return dist

def chi(x):
    if x < 0:
        return 1
    else:
        return 0


def DPC(distance,labels,K,t): # t=1%-2%
    distance_sort = []
    density = np.zeros(len(labels))

    for index_i in range(len(labels)):
        for index_j in range(index_i+1,len(labels)):
            distance_sort.append(distance[index_i, index_j])

        # compute optimal cutoff
        # distance_sort = np.array(distance_sort)
    # print(distance_sort)
    distance_sort.sort()
    cutoff = distance_sort[int(np.round(len(distance_sort) * t))]


    # computer density
    for index_i in range(len(labels)):
        distance_cutoff_i = distance[index_i] - cutoff
        for index_j in range(len(labels)):
            density[index_i] += chi(distance_cutoff_i[index_j])

    # distance_to_HDP
    dist = distance_to_HDP(distance,density)
    for i in range(len(density)):
        dist[i] = dist[i] * density[i]
    sort_points = sorted(enumerate(dist), key=lambda x: x[1], reverse=True)
    count = 0
    i = 0
    center_points = []
    while count < K:
        center_points.append(sort_points[i][0])
        count += 1
        i += 1
    return center_points


def FCM(X, c_clusters=3, m=2, eps=10):
    membership_mat = np.random.random((len(X), c_clusters))
    membership_mat = np.divide(membership_mat, np.sum(membership_mat, axis=1)[:, np.newaxis])

    max_iter = 30
    iter = 0
    while iter < max_iter:
        iter += 1
        working_membership_mat = membership_mat ** m
        Centroids = np.divide(np.dot(working_membership_mat.T, X),
                              np.sum(working_membership_mat.T, axis=1)[:, np.newaxis])

        n_c_distance_mat = np.zeros((len(X), c_clusters))
        for i, x in enumerate(X):
            for j, c in enumerate(Centroids):
                n_c_distance_mat[i][j] = np.linalg.norm(x - c, 2)

        new_membership_mat = np.zeros((len(X), c_clusters))

        for i, x in enumerate(X):
            for j, c in enumerate(Centroids):
                new_membership_mat[i][j] = 1. / np.sum((n_c_distance_mat[i][j] / n_c_distance_mat[i]) ** (2 / (m - 1)))
        if np.sum(abs(new_membership_mat - membership_mat)) < eps:
            break
        membership_mat = new_membership_mat
    return np.argmax(new_membership_mat, axis=1)

def feature_weight(X):
    W = []
    R = X.shape[2]
    for r in range(R):
        W.append(np.max(X[:,:,r]) - np.min(X[:,:,r]))
    W = W/np.sum(W)
    return W

def main():
    path = './data/'
    data_name = ['uWaveGesture']
    # 'lp1','lp5','wafer298','net534','cricket', 'JapaneseVowels','ArticularyWord', 'char', 'uWaveGesture','ArabicDigits'
    for file in data_name:
        filename = file + '.h5'
        filename = os.path.join(path, filename)
        f = h5py.File(filename, 'r')
        X = f['train_x'][:]
        # print(X.shape) # NxLxR
        y = f['train_y'][:]
        N = X.shape[0]
        R = X.shape[2]
        K = len(np.unique(y))

        print("对数据集%s进行FCFW聚类" % file)

        start = time.time()

        # 1. feature weight
        W = feature_weight(X)

        # 2.dtw distance
        PATH = './result/' + file + '_dtw_dist.npy'
        dtw_distance = np.load(PATH)
        print(dtw_distance.shape)
        if len(dtw_distance.shape) == 2:
            DTW = dtw_distance
        else:
            DTW = np.zeros((N, N))
            for i in range(1, N):
                for j in range(i):
                    for r in range(R):
                        DTW[i, j] = DTW[i, j] + dtw_distance[i, j, r]
                    DTW[j, i] = DTW[i,j]

        # 3. DPC based on DTW
        t = 0.02
        center_points = DPC(DTW, y, K, t)

        # fuzzy membership matrix
        Fdtw = np.zeros((N,K))
        for n in range(N):
            for k in range(K):
                if n == center_points[k]:
                    Fdtw[n,k] = 1
                    for i in range(k):
                        Fdtw[n, k] = 0
                    break
                Fdtw[n,k] = np.sqrt(1/DTW[n,center_points[k]])
            Fdtw[n,:] = Fdtw[n,:] / np.sum(Fdtw[n,:])

        # 4. SDB distance
        PATH = './result/' + file + '_dist.npy'
        sdb_distance = np.load(PATH)

        # 5. fuzzy membership based on SDB
        Fsdb = np.zeros((N, K))
        for r in range(R):
            Ftemp = np.zeros((N, K))
            for n in range(N):
                for k in range(K):
                    if n == center_points[k]:
                        Ftemp[n, k] = 1
                        for i in range(k):
                            Ftemp[n, k] = 0
                        break
                    if sdb_distance[n, center_points[k],r] == 0:
                        Ftemp[n, k] = 1
                        for i in range(k):
                            Ftemp[n, k] = 0
                        break
                    else:
                        Ftemp[n, k] = np.sqrt(1 / sdb_distance[n, center_points[k],r])
                Ftemp[n, :] = Ftemp[n, :] / np.sum(Ftemp[n, :])
            Fsdb += W[r]*Ftemp
        Fsdb /= R
        O = Fdtw + Fsdb

        # 6. FCM based on euclidean distance
        RI_value = 0
        purity_value = 0
        NMI_value = 0
        for i in range(10):# 10
            part = FCM(O, K)
            # print(part)

            RI_value += RandIndex(part, y)
            purity_value += Purity(part, y)
            NMI_value += NMI(part, y)
        end = time.time()
        print("总花费时间：",end-start)
        # print(RI_value/10, purity_value/10, NMI_value/10)

if __name__ == '__main__':
    main()


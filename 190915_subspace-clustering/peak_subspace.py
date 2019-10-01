import h5py
import numpy as np
import os
from distance_matrix import distance_matrix
from density_peak import densityPeakRNN
from cluster import series_to_centers
from update_subspace import getHA
from update_peak import update_peak

# 1、get Robot Execution Failures lp1-5 data
path = 'E:\\Jade\\time_series\\190808_MTS-clustering'
data_name = ['Robot Execution Failures lp1', 'Robot Execution Failures lp2', 'Robot Execution Failures lp3','Robot Execution Failures lp4','Robot Execution Failures lp5']
for file in data_name:
    filename = file + '.h5'
    filename = os.path.join(path,filename)
    f = h5py.File(filename,'r')
    X = f['train_x'][:]
    y = f['train_y'][:] # 1,2,3... np.array
    
    # 获取正常（没有变量权值）的距离矩阵
    PATH = './data/' + file + '_distance.npy'
    if os.path.exists(PATH):
        D = np.load(PATH)
    else:
        D = distance_matrix(X)
        np.save(PATH, D)

    # 2、初始化峰值和随机子空间
    K = len(np.unique(y))
    N = X.shape[0]
    L = X.shape[1]
    R = X.shape[2]

    # 2.1 初始化峰值
    center_label,density = densityPeakRNN(K,X,D)
    # 2.2 随机子空间
    np.random.seed(0)
    W = np.random.random((K,R))
    s = W.sum(axis=1)
    # 标准化，每一行相加等于1
    for i in range(K):
        W[i] = W[i] / s[i]
    
        
    # 迭代循环
    iter = 30
    for i in range(iter):
        # 3、样本分配到簇
        # 该步骤要用到变量子空间
        part = np.zeros(N)
        in_cluster = [[] for i in range(K)]
        for n in range(N):
            ck = series_to_centers(X,n,center_label,W)
            part[n] = ck
            in_cluster[ck].append(n)

        # 4、更新子空间和峰值
        # 4.1 更新子空间
        W = getHA(part,in_cluster,N,K,R,center_label,X)
        # 4.2 更新峰值
        center_label = update_peak(center_label,in_cluster,density,D)
    return part
    

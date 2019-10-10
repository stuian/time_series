import h5py
import numpy as np
import os
from distance_matrix import distance_matrix
from density_peak import densityPeakRNN
from cluster import series_to_centers
from update_subspace import getHA
from update_peak import update_peak
from evaluation import RandIndex
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
#从pyplot导入MultipleLocator类，这个类用于设置刻度间隔

# 1、get Robot Execution Failures lp1-5 data
path = 'E:\\Jade\\time_series\\190808_MTS-clustering'
# data_name = ['Robot Execution Failures lp1', 'Robot Execution Failures lp2', 'Robot Execution Failures lp3','Robot Execution Failures lp4','Robot Execution Failures lp5']
data_name = ['Robot Execution Failures lp5'] # ,'cricket_data'
for file in data_name:
    filename = file + '.h5'
    filename = os.path.join(path,filename)
    f = h5py.File(filename,'r')
    X = f['train_x'][:]
    y = f['train_y'][:] # 1,2,3... np.array

    print("%s数据集进行子空间聚类..." % file)
    
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
    # k:求某点是的密度，KNN的k值
    k = 10
    center_label,density = densityPeakRNN(k,K,X,D)
    # 2.2 随机子空间
    # np.random.seed(0)
    # W = np.random.random((K,R))
    # s = W.sum(axis=1)
    # # 标准化，每一行相加等于1
    # for i in range(K):
    #     W[i] = W[i] / s[i]
    W = np.ones((K,R))
    W = W / R
    
        
    # 迭代循环
    iter = 15
    RI = []
    x_axis = []
    for i in range(1,iter+1):
        # 3、样本分配到簇
        # 该步骤要用到变量子空间
        part = np.zeros(N)
        in_cluster = [[] for _ in range(K)]
        for n in range(N):
            ck = series_to_centers(X,n,center_label,W)
            part[n] = ck
            in_cluster[ck].append(n)

        # 4、更新子空间和峰值
        # 4.1 更新子空间
        W = getHA(part,in_cluster,N,K,R,center_label,X)
        print(W)
        # 4.2 更新峰值
        center_label = update_peak(center_label,in_cluster,density,D)

        # evaluation
        RI.append(RandIndex(part, y))
        x_axis.append(i)

    # save results
    PATH = './data/' + file + '_part.npy'
    np.save(PATH, part)

    # plot RI
    plt.figure()
    plt.xlim(1,iter)
    x_major_locator = MultipleLocator(1)
    ax = plt.gca()
    # ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    # 把x轴的主刻度设置为1的倍数
    plt.xlabel('iter')
    plt.ylabel('RI')
    plt.plot(x_axis,RI)
    picture_name = './data/' + file + '_RI-figure1.png'
    plt.savefig(picture_name)
    plt.show()

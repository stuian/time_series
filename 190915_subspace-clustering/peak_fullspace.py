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
data_name = ['Robot Execution Failures lp5']
for file in data_name:
    filename = file + '.h5'
    filename = os.path.join(path,filename)
    f = h5py.File(filename,'r')
    X = f['train_x'][:]
    y = f['train_y'][:] # 1,2,3... np.array

    print("%s数据集进行全空间聚类..." % file)

    # 获取正常（没有变量权值）的距离矩阵
    PATH = './data/' + file + '_distance.npy'
    if os.path.exists(PATH):
        D = np.load(PATH)
    else:
        D = distance_matrix(X)
        np.save(PATH, D)

    # 2、初始化峰值
    K = len(np.unique(y))
    N = X.shape[0]
    L = X.shape[1]
    R = X.shape[2]

    # k:求某点是的密度，KNN的k值
    k = 10
    center_label, density = densityPeakRNN(k, K, X, D)

    # 全空间
    W = np.ones((K,R))

    iter = 15
    RI = []
    x_axis = []
    for i in range(1, iter + 1):
        # 3、样本分配到簇
        part = np.zeros(N)
        in_cluster = [[] for _ in range(K)]
        print(in_cluster)
        for n in range(N):
            ck = series_to_centers(X, n, center_label, W)
            part[n] = ck
            in_cluster[ck].append(n)
        print(in_cluster)
        # 4、更新峰值
        center_label = update_peak(center_label, in_cluster, density, D)
        print(center_label)
        # evaluation
        RI.append(RandIndex(part, y))
        x_axis.append(i)
        print("iter%d" % i)

    # save results
    PATH = './data/' + file + '_full-part.npy'
    np.save(PATH, part)

    # plot RI
    plt.figure()
    plt.xlim(1, iter)
    x_major_locator = MultipleLocator(1)
    ax = plt.gca()
    # ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    # 把x轴的主刻度设置为1的倍数
    plt.xlabel('iter')
    plt.ylabel('RI')
    plt.plot(x_axis, RI)
    picture_name = './data/' + file + '_full-RI-figure.png'
    plt.savefig(picture_name)
    plt.show()
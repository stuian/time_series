import numpy as np
import os
import h5py
from DTW_distance_matrix import distance_matrix
from densityPeakRNN import densityPeakRNN
from densityPeak import densityPeak
from cluster import series_to_centers
# from updateW_HA import update_subspace
from updateW_MHA import update_subspace
from update_RNNpeak import update_peak
from evaluation import RandIndex
from evaluation import Purity
from evaluation import NMI
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

path = '../191020_subspace-clustering/data/'
data_name = ['Robot Execution Failures lp5']

for file in data_name:
    filename = file + '.h5'
    filename = os.path.join(path, filename)
    f = h5py.File(filename, 'r')
    X = f['train_x'][:]
    y = f['train_y'][:]  # 1,2,3... np.array

    print("%s数据集进行子空间聚类..." % file)
    print('数据集大小', X.shape)

    file = file + '_subspace'

    # 获取单变量距离矩阵
    PATH = './result/' + file + '_distance.npy'
    if os.path.exists(PATH):
        single_distance_between = np.load(PATH)
    else:
        single_distance_between = distance_matrix(X)
        np.save(PATH, single_distance_between)

    file = file + '_HA_MMD'

    # 2、初始化峰值和随机子空间
    K = len(np.unique(y))
    print('聚类簇数：', K)
    N = X.shape[0]
    R = X.shape[2]

    # 2.1 初始化峰值
    # k:求某点的密度中KNN的k值
    k = 30
    center_label, density = densityPeakRNN(k, K, X, single_distance_between)
    # center_label_2, density_2 = densityPeak(k,K,X,single_distance_between)
    print("center_label: ", center_label)
    # print("center_label: ", center_label_2)

    W = np.ones((K, R))
    W = W / R

    iter = 15
    RI = []
    purity = []
    nmi = []
    for i in range(1, iter + 1):
        # 3、样本分配到簇
        # 该步骤要用到变量子空间
        part = np.zeros(N)
        in_cluster = [[] for _ in range(K)]
        for n in range(N):
            ck = series_to_centers(single_distance_between, n, center_label, W)
            part[n] = ck
            in_cluster[ck].append(n)

        out_cluster = []
        for i in range(K):
            temp = []
            for j in range(K):
                if j != i:
                    temp = temp + in_cluster[j]
            out_cluster.append(temp)

        # 4、更新子空间和峰值
        # 4.1 更新子空间
        # W = update_subspace(in_cluster,out_cluster,center_label,single_distance_between) # HA
        # alphaSet = [3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8.5]
        # betaSet = [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.09, 0.1, 0.2, 0.5]
        alpha = 4
        beta = 0.01
        W = update_subspace(in_cluster, out_cluster, center_label, X, single_distance_between, alpha, beta)

        # 4.2 更新峰值
        # 子空间峰值更新
        center_label = update_peak(in_cluster,center_label,W,single_distance_between,density)

        # evaluation
        RI_value = RandIndex(part, y)
        purity_value = Purity(part, y)
        NMI_value = NMI(part, y)
        print(RI_value)
        RI.append(RI_value)
        purity.append(purity_value)
        nmi.append(NMI_value)

    # save results
    PATH = './result/' + file + '_part.npy'
    np.save(PATH, part)

    # save evaluation
    PATH = './result/' + file + '_RI.npy'
    np.save(PATH, RI)
    PATH = './result/' + file + '_purity.npy'
    np.save(PATH, purity)
    PATH = './result/' + file + '_NMI.npy'
    np.save(PATH, NMI)

    # plot RI
    plt.figure()
    filename = file + '_RI'
    plt.title(filename)
    plt.xlim(1, iter)
    x_major_locator = MultipleLocator(1)
    ax = plt.gca()
    # ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    # 把x轴的主刻度设置为1的倍数
    plt.xlabel('iter')
    plt.ylabel('RI')
    plt.plot(RI)
    picture_name = './pictures/' + filename + '-figure.png'
    plt.savefig(picture_name)
    plt.show()

    # plot purity
    plt.figure()
    filename = file + '_purity'
    plt.title(filename)
    plt.xlim(1, iter)
    x_major_locator = MultipleLocator(1)
    ax = plt.gca()
    # ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    # 把x轴的主刻度设置为1的倍数
    plt.xlabel('iter')
    plt.ylabel('purity')
    plt.plot(purity)
    picture_name = './pictures/' + filename + '-figure.png'
    plt.savefig(picture_name)
    plt.show()

    # plot NMI
    plt.figure()
    filename = file + '_NMI'
    plt.title(filename)
    plt.xlim(1, iter)
    x_major_locator = MultipleLocator(1)
    ax = plt.gca()
    # ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    # 把x轴的主刻度设置为1的倍数
    plt.xlabel('iter')
    plt.ylabel('NMI')
    plt.plot(nmi)
    picture_name = './pictures/' + filename + '-figure.png'
    plt.savefig(picture_name)
    plt.show()




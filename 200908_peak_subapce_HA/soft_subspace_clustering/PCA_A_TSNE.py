import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets._samples_generator import make_blobs
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import os
import h5py
from sklearn import decomposition
from sklearn.manifold import TSNE
from munkres import Munkres

def normal_data():
    # 簇1
    X, _ = make_blobs(n_samples=100, n_features=2, centers=[[-1, -1]],
                      cluster_std=[0.1], random_state=9) # [0, 0], [1, 1], [2, 2] , 0.2, 0.2, 0.2

    # plt.scatter(X[:, 0], X[:, 1], marker='o')  # 假设暂不知道y类别，不设置c=y，使用kmeans聚类
    np.random.seed(2)
    temp = np.random.randn(100,1)
    X = np.concatenate((X,temp),axis=1)
    y = [0 for i in range(100)]
    print(X.shape) # 1,2,3

    # 簇2

    # X2, y2 = make_blobs(n_samples=100, n_features=2, centers=[[1, 1]],
    #                   cluster_std=[0.2], random_state=8) # [0, 0], [1, 1], [2, 2] , 0.2, 0.2, 0.2
    X2 = X.copy() # 100
    y2 = [1 for _ in range(100)]
    X2[:,0] += 0.5
    np.random.seed(3)
    temp = np.random.randn(100,1)
    X2 = np.concatenate((temp,X2[:,1:2],X2[:,0:1]),axis = 1)
    print(X2.shape)
    # plt.scatter(X2[:50, 0], X2[:50, 1], marker='+')  # 假设暂不知道y类别，不设置c=y，使用kmeans聚类
    # plt.show()

    # 整体数据
    data = np.concatenate((X,X2[:50]),axis = 0)
    print(data.shape)
    labels = []
    for i in range(100):
        labels.append(y[i])
    for j in range(50):
        labels.append(y2[j])
    # print(labels)

    # 降维

    pca = PCA(n_components=2)
    A = pca.fit_transform(data)
    print(A.shape)
    # A = data[:,1:]

    # 真实情况
    A0 = []
    B0 = []
    A1 = []
    B1 = []
    for i in range(len(A)):
        if labels[i] == 0:
            A0.append(A[i][0])
            B0.append(A[i][1])
        elif labels[i] == 1:
            A1.append(A[i][0])
            B1.append(A[i][1])


    # plt.scatter(A0,B0, c = 'k')
    # plt.scatter(A1,B1, c = 'g')
    # plt.legend(["class1","class2"])
    # plt.show()

    #绘制k-means结果
    # x0 = X[label_pred == 0]
    # x1 = X[label_pred == 1]
    # x2 = X[label_pred == 2]
    # plt.scatter(x0[:, 0], x0[:, 1], c = "red", marker='o', label='label0')
    # plt.scatter(x1[:, 0], x1[:, 1], c = "green", marker='*', label='label1')
    # plt.scatter(x2[:, 0], x2[:, 1], c = "blue", marker='+', label='label2')
    # plt.xlabel('petal length')
    # plt.ylabel('petal width')
    # plt.legend(loc=2)

    # 预测情况
    y_pred_1 = KMeans(n_clusters=2, random_state=1).fit_predict(data[:,:2])
    print(y_pred_1)
    y_pred_2 = KMeans(n_clusters=2, random_state=1).fit_predict(data[:,1:])
    print(y_pred_2)
    y_ture = []
    for i in range(100):
        y_ture.append(y_pred_1[i])
    for i in range(100,150):
        y_ture.append(y_pred_2[i])
    A0 = []
    B0 = []
    A1 = []
    B1 = []
    for i in range(len(A)):
        if y_ture[i] == 0:
            A0.append(A[i][0])
            B0.append(A[i][1])
        elif y_ture[i] == 1:
            A1.append(A[i][0])
            B1.append(A[i][1])
    plt.scatter(A0,B0, c = 'k')
    plt.scatter(A1,B1, c = 'g')
    plt.legend(["class1","class2"])
    plt.show()

def MTS_toy_example_PCA():
    # data:lp1
    # 全空间
    path = './data/'
    file = 'lp5'
    filename = file + '.h5'
    filename = os.path.join(path, filename)
    f = h5py.File(filename, 'r')
    X = f['train_x'][:]
    print(X.shape)
    y = f['train_y'][:]
    K = len(np.unique(y))
    print(K)

    X = X[:,:,:2]

    # W = np.ones((K,X.shape[2]))

    X_new = []
    for i in range(len(X)):
        temp = np.zeros(X.shape[1])
        for r in range(X.shape[2]):
            temp += X[i,:,r]
        X_new.append(temp/X.shape[2])
    X_new = np.array(X_new)
    print("X_new的维度为：",X_new.shape)

    # 降维
    pca = PCA(n_components=2)
    A = pca.fit_transform(X_new)
    print(A.shape)

    # y_pred_full = np.load("./data/lp1_full_labels.npy")
    # A0 = []
    # B0 = []
    # A1 = []
    # B1 = []
    # A2 = []
    # B2 = []
    # A3 = []
    # B3 = []
    # for i in range(len(A)):
    #     if y_pred_full[i] == 0:
    #         A0.append(A[i][0])
    #         B0.append(A[i][1])
    #     elif y_pred_full[i] == 1:
    #         A1.append(A[i][0])
    #         B1.append(A[i][1])
    #     elif y_pred_full[i] == 2:
    #         A2.append(A[i][0])
    #         B2.append(A[i][1])
    #     elif y_pred_full[i] == 3:
    #         A3.append(A[i][0])
    #         B3.append(A[i][1])
    # plt.scatter(A0, B0, c='k')
    # plt.scatter(A1, B1, c='g')
    # plt.scatter(A2, B2, c='b')
    # plt.scatter(A3, B3, c='r')
    # plt.legend(["class1", "class2","class3","class4"])
    # plt.show()

    # print(y)
    A0 = []
    B0 = []
    A1 = []
    B1 = []
    A2 = []
    B2 = []
    A3 = []
    B3 = []
    A4 = []
    B4 = []
    for i in range(len(A)):
        if y[i] == 1:
            A0.append(A[i][0])
            B0.append(A[i][1])
        elif y[i] == 2:
            A1.append(A[i][0])
            B1.append(A[i][1])
        elif y[i] == 3:
            A2.append(A[i][0])
            B2.append(A[i][1])
        elif y[i] == 4:
            A3.append(A[i][0])
            B3.append(A[i][1])
        elif y[i] == 5:
            A4.append(A[i][0])
            B4.append(A[i][1])
    plt.scatter(A0, B0, c='k')
    plt.scatter(A1, B1, c='g')
    plt.scatter(A2, B2, c='b')
    plt.scatter(A3, B3, c='r')
    plt.scatter(A4, B4, c='m')
    plt.legend(["class1", "class2", "class3", "class4","class5"])
    plt.show()

def vectorized_MTS_TSNE():
    path = './data/'
    file = 'ArticularyWord'
    filename = file + '.h5'
    filename = os.path.join(path, filename)
    f = h5py.File(filename, 'r')
    X = f['train_x'][:]
    # print(X.shape)
    y = f['train_y'][:]

    # X = X[:,:,:5]

    # 数据向量化
    X_new = []
    for i in range(len(X)):
        temp = X[i].T
        temp = temp.flatten()
        X_new.append(temp)
    X_new = np.array(X_new)
    # X_pca = decomposition.TruncatedSVD(n_components=50).fit_transform(X_new)
    tsne = TSNE(n_components=2, perplexity=20 ,init='pca', random_state=503)

    X_tsne = tsne.fit_transform(X_new)

    '''嵌入空间可视化'''
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化

    # y_full_label = np.load("./data/"+ file + "_full_labels.npy")
    y_soft_label = np.load("./data/" + file + "_soft_labels.npy")
    #
    y_soft_label_new = best_map(y,y_soft_label)
    y_soft_label_new[162:184] = 8
    # y_soft_label_new[378:432][[y_soft_label_new[378:432] == 1]] = 18
    # y_soft_label_new[480:506][[y_soft_label_new[480:506]==6]] = 22
    # print(y)
    # print(y_soft_label_new)

    # y_full_label_new = best_map(y,y_full_label)
    # y_hard_label_new = best_map(y_true, y_hard_label)

    plt.figure(figsize=(5, 4))
    scatter = plt.scatter(X_norm[:, 0], X_norm[:, 1], c=y,cmap=plt.cm.nipy_spectral) # cmap="plasma";cmap=plt.cm.nipy_spectral;cmap=plt.cm.jet
    plt.xticks([])
    plt.yticks([])
    # plt.legend(*scatter.legend_elements())
    plt.show()


def best_map(L1,L2):
    #L1 should be the labels and L2 should be the clustering number we got
    Label1 = np.unique(L1)       # 去除重复的元素，由小大大排列
    nClass1 = len(Label1)        # 标签的大小
    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1,nClass2)
    G = np.zeros((nClass,nClass))
    for i in range(nClass1):
        ind_cla1 = L1 == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i,j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:,1]
    newL2 = np.zeros(L2.shape)
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2


if __name__ == '__main__':
    vectorized_MTS_TSNE()
import numpy as np
import os
import h5py
import time

def SGB(x,y):
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

def inner_product(a,b):
    product = np.dot(a,b)
    s = np.linalg.norm(a) * np.linalg.norm(b)
    if s == 0:
        product = 0
    else:
        product = product / s
    return 1-product # 越小越好


def main():
    path = './data/'
    dataname =["uWaveGestureLibrary"]# ["pen"]
    for file in dataname:
        begin_time = time.time()
        filename = file + '.h5'

        PATH = './result/'
        distancename = file + '_dist.npy'
        distance_path = os.path.join(PATH, distancename)

        filename = os.path.join(path, filename)
        f = h5py.File(filename, 'r')
        X = f['train_x'][:]
        print(X.shape)
        y = f['train_y'][:]  # 1,2,
        K = len(np.unique(y))
        print("聚类簇：%d" % K)

        n = X.shape[0]
        R = X.shape[2] # 变量数
        print(R)
        single_distance_between = np.zeros((n, n, R))
        print(file, '数据集开始计算距离相似矩阵')
        for i in range(1, n):
            for j in range(i):
                for r in range(R):
                    # single_distance_between[i, j, r],_ = fastdtw(X[i][:, r],X[j][:, r], dist=euclidean)
                    # single_distance_between[i,j,r] = inner_product(X[i][:, r], X[j][:, r])
                    # single_distance_between[i,j,r] = DTW(X[i][:, r], X[j][:, r])
                    single_distance_between[i, j, r] = SGB(X[i][:, r], X[j][:, r])
                    single_distance_between[j, i, r] = single_distance_between[i, j, r]

        print(single_distance_between.shape)
        print('完成计算！')
        np.save(distance_path, single_distance_between)
        print('已保存距离结果')
        end_time = time.time()
        print("time:",end_time-begin_time)

if __name__ == '__main__':
    main()

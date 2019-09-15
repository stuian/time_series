import numpy as np
import heapq
import h5py

# step1 distance between two multivariate series
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

def distance(X,Y):
    # DTWi
    distance = 0
    n = X.shape[1]
    for i in range(n):
        temp = DTW(X[:, i], Y[:, i])
        distance = distance + temp
    return distance

# step2 local density of every point with RNN
def KNN(X,x,k):
    """
    # 要不要一行一行的存储距离
    :param X:train_data
    :param x:第x个点
    :return:knn列表
    """
    curr_colum = []
    for i in range(X.shape[0]):
        if i == x:
            curr_colum.append(float('inf'))
        else:
            curr_colum.append(distance(X[x],X[i]))
    temp = map(curr_colum.index, heapq.nsmallest(k, curr_colum))
    return list(temp)

def RNN(X,x,k):
    count = 0
    for y in range(X.shape[0]):
        if y != x:
            if x in KNN(X,y,k):
                count += 1
    return count

def densityPeakRNN(k,X):
    # X:样本数量,样本长度,变量数目
    # return centerlabel
    n = X.shape[0]
    rnn = np.zeros(n) # 求得每个序列的反近邻数量
    for i in range(n):
        # sample = X[i]
        rnn[i] = RNN(X,i,k)
    # print(rnn)
    

if __name__ == '__main__':
    k = 3
    file = h5py.File('E:\\Jade\\time_series\\190808_MTS-clustering\\cricket_data.h5', 'r')
    X = file['train_x'][:]
    y = file['train_y'][:]
    print(len(np.unique(y))) # 12
    densityPeakRNN(k,X)
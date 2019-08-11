"""
author:Jade
date:190726
refer:
"""

import data.load_data
import numpy as np
import heapq
import matplotlib.pyplot as plt

def RNN_DBSCAN(X,k,size):
    """
    :param X:train_X
    :param k: k个最相似
    :param size:样本个数
    :return:
    """
    assign = np.zeros(size) # 0 denotes unclassified
    cluster = 1
    for x in range(size):
        if assign[x] == 0:
            if ExpandCluster(X,x,cluster,assign):
                cluster = cluster + 1
    ExpandClusters(X,assign)
    return assign

def ExpandCluster(X,x,cluster,assign):
    if len(RNN(X,x)) < k :
        assign[x] = -1 # noise
        return False
    else:
        assign[x] = cluster
        neighbors = Neighborhood(X,x) # list
        for i in neighbors:
            assign[i] = cluster
        count = 0
        while count < len(neighbors):
            y = neighbors[count]
            if len(RNN(X,y)) >= k:
                tempneighbors = Neighborhood(X,y)
                for z in tempneighbors:
                    if assign[z] == 0:
                        neighbors.append(z)
                        assign[z] = cluster
                    elif assign[z] == -1: #expand
                        assign[z] = cluster
            count += 1
        return True

def ExpandClusters(X,assign):
    for x in range(size):
        if assign[x] == -1 :
            neighbors = KNN(X,x)
            mincluster = -1
            mindist = float('inf')
            for n in neighbors:
                cluster = assign[n]
                d = dist(X[x],X[n])
                if RNN(X,n) >= k and d <= density(X,cluster,assign) and d < mindist:
                    mincluster = cluster
                    mindist = d

def dist(A,B):
    return np.sqrt(sum(np.power(A-B,2)))

def KNN(X,x):
    """
    :param X:train_data
    :param x:第x个点
    :param NN: 距离矩阵
    :return:
    """
    curr_colum = []
    for i in range(size):
        if i == x:
            curr_colum.append(float('inf'))
        else:
            curr_colum.append(dist(X[x],X[i]))
    temp = map(curr_colum.index, heapq.nsmallest(k, curr_colum))
    return list(temp)

def RNN(X,x):
    rnn = []
    for y in range(size):
        if y != x:
            if x in KNN(X,y):
                rnn.append(y)
    return rnn

def density(X,cluster,assign):
    cluster_set = []
    for i in range(size):
        if assign[i] == cluster:
            cluster_set.append(i)
    max = -1
    for i in range(1,len(cluster)):
        for j in range(i):
            temp = dist(X[i], X[j])
            if temp > max:
                max = temp
    return max

def Neighborhood(X,x):
    """
    x的k个邻近点和x的k个反邻近点（并且该反邻近点是中心点）
    :param x:
    :param k:
    :return:
    """
    neighborhoods = KNN(X,x)
    rneighbors = RNN(X,x)
    for y in rneighbors:
        if len(RNN(X,y)) >= k and y not in neighborhoods:
            neighborhoods.append(y)
    return neighborhoods

if __name__ == '__main__':
    train_data,label = data.load_data.concat_data()
    k = 5
    size = train_data.shape[0] # 训练样本数量
    result = RNN_DBSCAN(train_data,k,size)
    plt.figure(figsize=(12, 9), dpi=80)
    plt.scatter(train_data, label, c=result, marker='.')
    plt.show()

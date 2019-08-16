"""
author:Jade
date:190802
theme:基于DTW的矩阵相似
"""

import numpy as np
from txt_to_matrix import two_matrix_samples

def DTW(s1,s2):
    """
    计算两个向量的DTW值
    :param s1: 向量1;list和array类型都可以
    :param s2: 向量2
    :return:
    """
    DTW = {}
    for i in range(len(s1)):
        DTW[(i,-1)] = float('inf')
    for i in range(len(s2)):
        DTW[(-1,i)] = float('inf')
    DTW[(-1,-1)] = 0
    for i in range(len(s1)):
        for j in range(len(s2)):
            dist = (s1[i] - s2[j])**2
            DTW[(i,j)]= dist + min(DTW[(i-1,j)],DTW[(i,j-1)],DTW[(i-1,j-1)])
    return np.sqrt(DTW[(len(s1)-1,len(s2)-1)])

# 判断矩阵的相似性
def matrix_DTWi(X,Y):
    distance = 0
    n = X.shape[1]
    for i in range(n):
        temp = DTW(X[:,i],Y[:,i])
        distance = distance + temp
    return distance

def matrix_DTWd(X,Y):
    m = X.shape[0]
    n = Y.shape[0]
    matrix_DTW = {}
    for i in range(m):
        matrix_DTW[(i,-1)] = float('inf')
    for i in range(n):
        matrix_DTW[(-1,i)] = float('inf')
    matrix_DTW[(-1,-1)] = 0
    for i in range(m):
        for j in range(n):
            matrix_DTW[(i,j)] = dist(X[i,:],Y[j,:]) + min(matrix_DTW[(i-1,j)],matrix_DTW[(i,j-1)],matrix_DTW[(i-1,j-1)])
    return np.sqrt(matrix_DTW[(m-1,n-1)])

def dist(a,b):
    return np.sqrt(sum(np.power(a-b,2)))


if __name__ == '__main__':
    X1,X2 = two_matrix_samples()
    #print(X1.shape,X2.shape)  # (137, 6) (143, 6)
    print(DTW(X1[:,0],X2[:,0]))
"""
date:2019-7-17
author:Jade
theme:a fast similar measure and CPA(constraint promotion approach)
"""

import numpy as np
import os
import heapq
import DBSCAN
import random
import math
from sklearn import metrics
import matplotlib.pyplot as plt
import time

def load_data(path):
    """
    :param path:所读取文件的路径
    :return:
    """
    file_list = os.listdir(path)
    count = 0
    for file in file_list:  # 127个
        final_path = os.path.join(path, file)
        if count == 0:
            data = np.genfromtxt(final_path, delimiter=' ')
        else:
            temp = np.genfromtxt(final_path, delimiter=' ')
            data = np.vstack((data, temp))
        count += 1
    return data

def concat_data():
    """
    wafer dataset
    :return: X,y
    """
    abnormal_path = "./data/wafer/abnormal"
    normal_path = "./data/wafer/normal"
    abnormal_data = load_data(abnormal_path)
    abnormal_data = np.c_[abnormal_data,np.ones(len(abnormal_data))]

    normal_data = load_data(normal_path)
    normal_data = np.c_[normal_data,np.zeros(len(normal_data))]

    data = np.vstack((abnormal_data,normal_data))
    # print(data.shape) #(2756, 7)
    # print(data[0])    #[  2. -11.  -1.   3.  24.  10.   1.]
    return data[:,:-1],data[:,-1]

def Revers_nearest_neighbor(query,NN):
    """
    Each one in RNN(q) treat q as its nearest neighbor
    :param query:请求寻找rnn的data point
    :param NN:距离矩阵
    :return:query的反最近邻
    """
    RNN = []
    for i in range(NN.shape[0]):
        if i!= query and min(NN[i]) == NN[i,query]:
            RNN.append(i)
    return RNN

def Auto_auxi(MustLink,CannotLink,nn):
    """
    根据人工给出的限制用CAP策略进行拓展
    :param MustLink:[[1,2],[1.2]...]
    :param CannotLink:
    :param nn: data之间的距离矩阵
    :return:autoMust;autoCannot
    """
    x1 = MustLink.shape[0]
    x2 = CannotLink.shape[0]
    #queries_Must = np.zeros((1,2*x1))
    #queries_Cannot = np.zeros((1,2*x2))
    # for i in range(x1):
    autoMust = []
    autoCannot = []
    for i in range(x1):
        rnn1 = Revers_nearest_neighbor(MustLink[i][0],nn)
        rnn2 = Revers_nearest_neighbor(MustLink[i][1],nn)
        if len(rnn1)>0:
            for j in range(len(rnn1)):
                autoMust.append([rnn1[j],MustLink[i][0]])
        if len(rnn2)>0:
            for j in range(len(rnn2)):
                autoMust.append([rnn2[j],MustLink[i][1]])
    for i in range(x2):
        rnn1 = Revers_nearest_neighbor(CannotLink[i][0], nn)
        rnn2 = Revers_nearest_neighbor(CannotLink[i][1], nn)
        if len(rnn1) > 0:
            for j in range(len(rnn1)):
                autoCannot.append([rnn1[j], CannotLink[i][1]])
        if len(rnn2) > 0:
            for j in range(len(rnn2)):
                autoCannot.append([rnn2[j], CannotLink[i][0]])
    return np.array(autoMust),np.array(autoCannot)

def lb_keogh(s1,s2):
    """
    :param s1:vec1
    :param s2: vec2
    :return: lb值
    """
    r = len(s1) // 7 #？为什么除以7
    LB_sum = 0
    n = len(s2)
    for ind, i in enumerate(s1):
        lower_bound = min(s2[(ind - r if ind - r >= 0 else 0):(ind + r if ind + r <= n else n)])
        upper_bound = max(s2[(ind - r if ind - r >= 0 else 0):(ind + r if ind + r <= n else n)])
        if i > upper_bound:
            LB_sum += (i - upper_bound) ** 2
        elif i < lower_bound:
            LB_sum += (i - lower_bound) ** 2
    return np.sqrt(LB_sum)

def ud_ED(data1,data2):
    """
    欧几里得距离
    :param data1:vec1
    :param data2: vec2
    :return:ED距离
    """
    return np.sqrt(sum(np.power(data1-data2,2)))

def aDTW(s1,s2,beta):
    LB = lb_keogh(s1, s2)  # lower bound
    UB = ub_ED(s1, s2)  # upper bound
    return LB + beta * (UB - LB)

def aDTW_calculate(C,D,beta):
    """
    :param C:[[],[]]
    :param D:[[],...]
    :param beta:
    :return:aDTW矩阵
    """
    k = C.shape[0]
    n = D.shape[0]
    aDTW = np.zeros((k,n))
    for i in range(k):
        for j in range(n):
            LB = lb_keogh(C[i],D[j]) #lower bound
            UB = ub_ED(C[i],D[j]) #upper bound
            aDTW[i,j] = LB + beta*(UB-LB)
    return aDTW

def Bound_Calculate(C,D,Uorl):
    k = C.shape[0]
    n = D.shape[0]
    bound = np.zeros((k,n))
    for i in range(k):
        for j in range(n):
            if Uorl == 1:
                bound[i,j] = ub_ED(C[i],D[j])
            elif Uorl == 2:
                bound[i,j] = lb_keogh(C[i],D[j])
    return bound

def sigmoid(X):
    return 1.0/(1+exp(-X))

def tDTW_calculate(s1,s2):
    # 是s1,s2为两列时间序列数据
    DWT = {}
    for i in range(len(s1)):
        DWT[(i, -1)] = float('inf')
    for i in range(len(s2)):
        DWT[(-1, i)] = float('inf')
    DWT[(-1, -1)] = 0
    for i in range(len(s1)):
        for j in range(len(s2)):
            dist = (s1[i] - s2[j]) ** 2
            DWT[(i, j)] = dist + min(DWT[(i - 1, j)], DWT[(i, j - 1)], DWT[(i - 1, j - 1)])
    return np.sqrt(DWT[(len(s1) - 1, len(s2) - 1)])

def Purity(y_pred,labels):
    """
    :param train_labels:y_pred
    :param labels:
    :return:
    """
    length = len(labels)
    classIndex = np.unique(y_pred)
    classNum = len(classIndex)
    maxSum = 0
    for i in range(classNum):
        currIndex = classIndex[i]
        currDataIndex = []
        for j in range(length):
            if y_pred[j] == currIndex:
                currDataIndex.append(j)
        for k in currDataIndex:
            if y_pred[k] == labels[k]:
                maxSum += 1
    return maxSum/length

def RandIndex(y_pred,labels):
    """
    :param y_pred:predict label
    :param labels: true label
    :return:
    """
    length = len(labels)
    TP,TN ,FP,FN = 0,0,0,0
    for k1 in range(length-1):
        for k2 in range(k1+1,length):
            if y_pred[k1]== y_pred[k2] and labels[k1]==labels[k2]:
                # 本身是同一类，且被分到了同一类
                TP = TP + 1
            elif y_pred[k1] != y_pred[k2] and labels[k1]!=labels[k2]:
                # 本身不是同一类，且被分到了不同类
                TN = TN +1
            elif y_pred[k1] == y_pred[k2] and labels[k1] != labels[k2]:
                # 不同类被分到同一类
                FP = FP +1
            elif y_pred[k1] != y_pred[k2] and labels[k1] == labels[k2]:
                # 同一类被分到不同类
                FN = FN +1
    return (TP+TN)/(TP+FP+FN+TN)

def NMI(A,B):
    """
    https://blog.csdn.net/chengwenyao18/article/details/45913891
    :param A:np.array
    :param B:np.array
    :return:
    """
    #样本点数
    total = len(A)
    A_ids = set(A)
    B_ids = set(B)
    #互信息计算
    MI = 0
    eps = 1.4e-45
    for idA in A_ids:
        for idB in B_ids:
            idAOccur = np.where(A==idA)
            idBOccur = np.where(B==idB)
            idABOccur = np.intersect1d(idAOccur,idBOccur) #取两者公共包含的数
            px = 1.0*len(idAOccur[0])/total
            py = 1.0*len(idBOccur[0])/total
            pxy = 1.0*len(idABOccur)/total
            MI = MI + pxy*math.log(pxy/(px*py)+eps,2) #为什么要加eps
    # 标准化互信息
    Hx = 0
    for idA in A_ids:
        idAOccurCount = 1.0*len(np.where(A==idA)[0])
        Hx = Hx - (idAOccurCount/total)*math.log(idAOccurCount/total+eps,2)
    Hy = 0
    for idB in B_ids:
        idBOccurCount = 1.0*len(np.where(B==idB)[0])
        Hy = Hy - (idBOccurCount/total)*math.log(idBOccurCount/total+eps,2)
    MIhat = 2.0*MI/(Hx+Hy)
    return MIhat

def K_means_fDTW(C,train_data,beta,label,needSigma,useBound):
    """
    :param C:
    :param train_data:
    :param beta:
    :param label:
    :param needSigma:update DTW or not
    :param useBound: 0(use our aDTW), 1(use UB as aDTW), 2(use LB as aDTW)
    :return:clustering result and randindex
    """
    iter = 15
    k = C.shape[0]
    n = train_data.shape[0]
    if useBound == 0:
        aDTW = aDTW_calculate(C,train_data,beta)
    else:
        aDTW = Bound_Calculate(C,train_data,useBound)

    index = np.zeros(n)

    # δ
    if needSigma == 1:
        aDTW_std = np.std(aDTW)
        sigma = sigmoid(aDTW_std)*aDTW_std
        sigma = sigma*2
    else:
        sigma = 0

    for i in range(iter):
        for p in range(n):
            curr_colum = aDTW[:,p]
            temp = map(list(curr_colum).index,heapq.nsmallest(2,curr_colum))
            temp = list(temp)
            index1 = temp[0]
            min1 = curr_colum[index1]
            index2 = temp[1]
            min2 = curr_colum[index2]
            if abs(min2 - min1) >= sigma:
                index[p] = index1
            else:
                min1 = tDTW_calculate(C[index1],train_data[p])
                min2 = tDTW_calculate(C[index2],train_data[p])
                aDTW[index1,p] = min1
                aDTW[index2,p] = min2
                if min1 < min2:
                    index[p] = index1 # 分类标记
                else:
                    index[p] = index2
        # 更新中心点
        for k1 in range(k):
            clust_ind = []
            for ind,j in enumerate(index):
                if j == k1:
                    clust_ind.append(ind)
            temp = np.zeros(C.shape[1])
            for k2 in range(clust_index):
                temp = temp + train_data[k2]
            temp = temp/len(clust_ind)
            C[k1] = temp # 更新
        aDTW = aDTW_calculate(C,train_data,beta)

    return index,RandIndex(index,label)

def K_means_fDTW_auxi(C,train_data,beta, label,needSigma,useBound, MustLink,CannotLink):
    # 数据的10%作为pairwise constraint
    iter = 15
    k = C.shape[0]
    n = train_data.shape[0]
    if useBound == 0:
        aDTW = aDTW_calculate(C,train_data, beta)
    else:
        aDTW = Bound_Calculate(C, train_data, useBound)

    index = np.zeros(n)

    if needSigma == 1:
        aDTW_std = np.std(aDTW)
        sigma = sigmoid(aDTW_std) * aDTW_std
        sigma = sigma * 2  # δ?
    else:
        sigma = 0
    classes = {}
    for i in range(k):
        classes[i] = [] #？
    for i in range(iter):
        for p in range(n):
            curr_colum = aDTW[:, p]
            temp = map(list(curr_colum).index, heapq.nsmallest(2, curr_colum))
            temp = list(temp)
            index1 = temp[0]
            min1 = curr_colum[index1]
            index2 = temp[1]
            min2 = curr_colum[index2]
            changed = False
            if abs(min2 - min1) >= sigma:
                # add CannotLink
                for s in range(CannotLink.shape[0]):
                    if (p == CannotLink[s,0] and CannotLink[s,1] in classes[index1]) \
                            or (p == CannotLink[s,1] and CannotLink[s,0] in classes[index1]):
                        changed = True
                        break
                if changed:
                    changed = False
                    index[p] = index2
                else:
                    index[p] = index1
            else:
                min1 = tDTW_calculate(C[index1], train_data[p])
                min2 = tDTW_calculate(C[index2], train_data[p])
                aDTW[index1, p] = min1
                aDTW[index2, p] = min2
                if min1 < min2:
                    for s in range(CannotLink.shape[0]):
                        if (p == CannotLink[s, 0] and CannotLink[s, 1] in classes[index1]) \
                                or (p == CannotLink[s, 1] and CannotLink[s, 0] in classes[index1]):
                            changed = True
                            break
                    if changed:
                        changed = False
                        index[p] = index2
                    else:
                        index[p] = index1
                else:
                    for s in range(CannotLink.shape[0]):
                        if (p == CannotLink[s, 0] and CannotLink[s, 1] in classes[index2]) \
                                or (p == CannotLink[s, 1] and CannotLink[s, 0] in classes[index2]):
                            changed = True
                            break
                    if changed:
                        changed = False
                        index[p] = index1
                    else:
                        index[p] = index2
            classes[index[p]].append(p)
            # add MustLink
            for u in range(MustLink.shape[0]):
                if p == MustLink[u,0]:
                    index[MustLink[u,1]] = index[p]
                    classes[index[p]].append(MustLink[u,1])
                elif p == MustLink[u,1]:
                    index[MustLink[u,0]] = index[p]
                    classes[index[p]].append(MustLink[u,0])

        # update Center point
        for k1 in range(k):
            clust_ind = []
            for ind, j in enumerate(index):
                if j == k1:
                    clust_ind.append(ind)
            temp = np.zeros(C.shape[1])
            for k2 in range(clust_index):
                temp = temp + train_data[k2]
            temp = temp / len(clust_ind)
            C[k1] = temp  # 更新
        aDTW = aDTW_calculate(C, train_data, beta)
    return index,RandIndex(index,label)

def K_means_trueDTW(C,train_data,label):
    k = C.shape[0]
    n = train_data.shape[n]
    iter = 15
    index = np.zeros(n)
    tDTW = np.zeros((k,n))
    for i in range(k):
        for j in range(n):
            tDTW[i,j] = tDTW_calculate(C[i],train_data[j])
    for i in range(iter):
        for p in range(n):
            temp = np.float("inf")
            for q in range(k):
                if tDTW[q,p] < temp:
                    temp = tDTW[q,p]
                    index[p] = q
        # update center points
        for k1 in range(k):
            clust_ind = []
            for ind, j in enumerate(index):
                if j == k1:
                    clust_ind.append(ind)
            temp = np.zeros(C.shape[1])
            for k2 in range(clust_index):
                temp = temp + train_data[k2]
            temp = temp / len(clust_ind)
            C[k1] = temp  # 更新
        for s in range(k):
            for t in range(n):
                tDTW[s,t] = tDTW_calculate(C[s],train_data[t])
    return index,RandIndex(index,label)

def K_means_trueDTW_auxi(C,train_data,label):
    k = C.shape[0]
    n = train_data.shape[0]
    iter = 15
    index = np.zeros(n)
    classes = {}
    for i in range(k):
        classes[i] = []
    for i in range(iter):
        tDTW = np.zeros((k,n))
        for s in range(k):
            for t in range(t):
                tDTW[s,t] = tDTW_calculate(C[s],train_data[t])
        for p in range(n):
            temp = np.float('inf')
            changed = False
            for q in range(k):
                if tDTW[q,p] < temp:
                    # CannotLink
                    for s in range(CannotLink.shape[0]):
                        if (p == CannotLink[s, 0] and CannotLink[s, 1] in classes[q]) \
                                or (p == CannotLink[s, 1] and CannotLink[s, 0] in classes[q]):
                            changed = True
                            break
                    if changed:
                        changed = False
                        temp = tDTW[q,p]
                        index[p] = q
            classes[index[p]].append(p)
            # MustLink
            for u in range(MustLink.shape[0]):
                if p == MustLink[u,0]:
                    index[MustLink[u,1]] = index[p]
                    classes[index[p]].append(MustLink[u,1])
                elif p == MustLink[u,1]:
                    index[MustLink[u,0]] = index[p]
                    classes[index[p]].append(MustLink[u,0])
        # update Center point
        for k1 in range(k):
            clust_ind = []
            for ind, j in enumerate(index):
                if j == k1:
                    clust_ind.append(ind)
            temp = np.zeros(C.shape[1])
            for k2 in range(clust_index):
                temp = temp + train_data[k2]
            temp = temp / len(clust_ind)
            C[k1] = temp  # 更新
    return index, RandIndex(index, label)

# 导入DBSCAN函数,DBSCAN(ED)

def DBSCAN_DTW(dataSet,eps,minPts,label):
    """
    :param dataSet:
    :param eps:k-dist function (refer[50])
    :param minPts:根据参考文献，设置为5
    :return:
    """
    nPoints = dataSet.shape[0]
    vPoints = DBSCAN.visitlist(count=nPoints)
    k = -1
    # 初始所有数据标记为-1
    C = [-1 for i in range(nPoints)]
    while vPoints.unvisitednum > 0:
        P = random.choice(vPoints.unvisitedlist)
        vPoints.visit(P)
        # N：求P的邻域
        N = [i for i in range(nPoints) if tDTW_calculate(dataSet[i], dataSet[P]) <= eps]
        if len(N) >= minPts:  # P的邻域里至少有minPts个对象
            # 创建新簇，把P添加到新簇
            k += 1
            C[P] = k
            for pl in N:
                if pl in vPoints.unvisitedlist:
                    vPoints.visit(pl)
                    M = [i for i in range(nPoints) if tDTW_calculate(dataSet[i], dataSet[pl]) <= eps]
                    if len(M) >= minPts:
                        for i in M:
                            if i not in N:
                                N.append(i)  # N长度增加，循环次数也增多了
                    if C[pl] == -1:
                        C[pl] = k
        else:
            C[P] = -1
    return C,RandIndex(index, label)

if __name__ == '__main__':
    K_means_fDTW(C, X, 0.3, y, 0, 1)
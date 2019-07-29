"""
date:2019-07-05
author:Jade
content:对时间序列进行分类
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

train = np.genfromtxt('train.txt',delimiter='\t')
test = np.genfromtxt('test.txt',delimiter='\t')

def DWTDistance_Window(s1,s2,w):
    DWT = {}
    m = len(s1)
    n = len(s2)
    w = max(w,abs(m-n))
    for i in range(-1,m):
        for j in range(-1,n):
            DWT[(i,j)] = float('inf')
    DWT[(-1,-1)] = 0
    for i in range(m):
        for j in range(max(0,i-w),min(n,i+w)):
            dist = (s1[i]-s2[j])**2
            DWT[(i,j)] = dist + min(DWT[(i-1,j)],DWT[(i,j-1)],DWT[(i-1,j-1)])
    return np.sqrt(DWT[(m-1,n-1)])

def LB_Keogh(s1,s2,r):
    LB_sum = 0
    n = len(s2)
    for ind,i in enumerate(s1):
        # r表示边界范围
        lower_bound = min(s2[(ind-r if ind-r >= 0 else 0):(ind+r if ind+r <=n else n)])
        upper_bound = max(s2[(ind-r if ind-r >= 0 else 0):(ind+r if ind+r <=n else n)])
        if i>upper_bound:
            LB_sum += (i-upper_bound)**2
        elif i<lower_bound:
            LB_sum += (i-lower_bound)**2
    return np.sqrt(LB_sum)

def Knn(train,test,w):
    preds = []
    for ind,i in enumerate(test[:,:-1]):
        min_dist = float('inf')
        key = None
        for ind_j,j in enumerate(train[:,:-1]):
            if LB_Keogh(i,j,5) < min_dist:
                cur_distance = DWTDistance_Window(i,j,w)
                if cur_distance < min_dist:
                    min_dist = cur_distance
                    key = ind_j
        preds.append(train[key][-1])
    return classification_report(test[:,-1],preds)

print(Knn(train,test,4))
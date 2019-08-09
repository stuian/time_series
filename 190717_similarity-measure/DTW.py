"""
date:2019-07-05
author:Jade
content:判断两个时间序列是否相似
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1、O(m*n)运算速度,求距离最小值
def DWTDistance(s1,s2):
    DWT = {}
    for i in range(len(s1)):
        DWT[(i,-1)] = float('inf')
    for i in range(len(s2)):
        DWT[(-1,i)] = float('inf')
    DWT[(-1,-1)] = 0
    for i in range(len(s1)):
        for j in range(len(s2)):
            dist = (s1[i] - s2[j])**2
            DWT[(i,j)]= dist + min(DWT[(i-1,j)],DWT[(i,j-1)],DWT[(i-1,j-1)])
    return np.sqrt(DWT[(len(s1)-1,len(s2)-1)])

# 2、增加w窗口值，提高速度
# 当i和j相距较远，选择不计算，提高速度

def DWTDistance_window(s1,s2,w):
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


# 3、下界方法,O(n)
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

if __name__ == '__main__':
    # data
    # ts_a = [1,2,3,4,5,6] # m = 6
    # ts_b = [2,2,2,2] # n = 4

    x = np.linspace(0, 50, 100)  # 0-50之间取100个数
    ts_a = pd.Series(3.1 * np.sin(x / 1.5) + 3.5)
    ts_b = pd.Series(2.2 * np.sin(x / 3.5 + 2.4) + 3.2)

    print(DWTDistance(ts_a, ts_b))
    print(DWTDistance_window(ts_a, ts_b, 10))
    print(LB_Keogh(ts_a, ts_b, 20))
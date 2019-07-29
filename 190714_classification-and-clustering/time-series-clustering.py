import random
import numpy as np
import matplotlib.pyplot as plt

train = np.genfromtxt('train.txt',delimiter='\t') # 时间序列长度大概为60
test = np.genfromtxt('test.txt',delimiter='\t')
data = np.vstack((train[:,:-1],test[:,:-1]))

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

def K_Means(data,num_clust,num_iter,w):
    centroids = {}
    data = list(data)
    temp = random.sample(data,num_clust)
    for ind,i in enumerate(temp):
        centroids[ind] = i
    counter = 0
    for _ in range(num_iter):
        counter += 1
        print(counter)
        assignments = {}
        for ind,i in enumerate(data):
            min_dist = float('inf')
            closest_clust = None
            for c_ind,j in enumerate(centroids.values()):
                if LB_Keogh(i,j,5) < min_dist:
                    cur_dist = DWTDistance_window(i,j,w)
                    if cur_dist < min_dist:
                        min_dist = cur_dist
                        closest_clust = c_ind
            if closest_clust in assignments:
                assignments[closest_clust].append(ind)
            else:
                assignments[closest_clust] = []

        #更新聚类效果
        for key in assignments:
            clust_sum = 0
            for k in assignments[key]:
                clust_sum = clust_sum + data[k]
            centroids[key] = [m/len(assignments[key]) for m in clust_sum]
    return centroids

centroids = K_Means(data,4,10,4)
# {0: [-0.8804256192381439, -1.0270818225288658, -0.8583330434432989, -0.8842706313608246,...
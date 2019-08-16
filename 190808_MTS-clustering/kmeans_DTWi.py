import numpy as np
import time
from evaluation import Purity
from evaluation import RandIndex
from evaluation import NMI
import h5py


def init_random_medoids(X,y,k):
    centroids = np.zeros((k, X.shape[1], X.shape[2]))
    for i in range(k):
        count = 0
        for j in range(len(y)):
            if y[j] == i + 1:
                centroids[i]  = centroids[i] + X[j]
                count += 1
        centroids[i] = centroids[i] / count
        # centroid = X[np.random.choice(range(n_samples))]
        # centroids[i] = centroid
    return centroids

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

def DTWi(X,Y):
    distance = 0
    n = X.shape[1]
    for i in range(n):
        temp = DTW(X[:,i],Y[:,i])
        distance = distance + temp
    return distance

def update_centroids(k,X,y_pred,centroids):
    # update Center point
    for k1 in range(k):
        clust_ind = []
        for ind, j in enumerate(y_pred):
            if j == k1+1:
                clust_ind.append(ind)
        temp = np.zeros(centroids[0].shape)
        for k2 in clust_ind:
            temp = temp + X[k2]
            temp = temp / len(clust_ind)  # 0/0
        centroids[k1] = temp  # 更新
    return centroids

def kmeans_DTWi(X,y):
    k = len(np.unique(y))
    m = X.shape[0]

    y_pred = np.zeros(m)
    centroids = init_random_medoids(X,y,k)

    clusterChanged = 0
    start = time.time()
    while clusterChanged < 50:
        # clusterChanged = False
        for i in range(m):
            minDist = np.inf
            minIndex = -1
            for j in range(k):
                distJI = DTWi(centroids[j],X[i])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j+1
            if y_pred[i] != minIndex:
                # clusterChanged = True  # 一直到收敛，不在更新，则迭代完成
                y_pred[i] = minIndex
        clusterChanged += 1
        centroids = update_centroids(k,X,y_pred,centroids)
    end = time.time()
    return float(end-start),y_pred,RandIndex(y_pred,y),Purity(y_pred,y),NMI(y_pred,y)


def main():
    # 1、get Robot Execution Failures lp1-5 data
    data_name = ['Robot Execution Failures lp1', 'Robot Execution Failures lp2', 'Robot Execution Failures lp3','Robot Execution Failures lp4','Robot Execution Failures lp5']
    count = 1
    for file in data_name:
        filename = file + '.h5'
        f = h5py.File(filename,'r')
        X = f['train_x'][:]
        y = f['train_y'][:] # 1,2,3... np.array
        cost_time,y_pred,randindex,purity,nmi= kmeans_DTWi(X,y)
        newfilename = str(count) + '.npy'
        count += 1
        np.save(newfilename,y_pred)
        print(file,"数据集聚类的时间为%.2f秒,RI值为%.2f,purity值为%.2f,nmi值为%.2f" % (cost_time,randindex,purity,nmi))

    # 2、

if __name__ == '__main__':
    main()




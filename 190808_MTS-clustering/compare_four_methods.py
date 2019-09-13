from kmeans_DTWi import kmeans_DTWi
from kmeans_DTWd import kmeans_DTWd
from kmeans_aDTWi import kmeans_aDTWi
from kmeans_dot import kmeans_dot
import h5py
import numpy as np

def init_random_medoids(X,k):
    centroids = np.zeros((k, X.shape[1], X.shape[2]))
    for i in range(k):
        centroid = X[np.random.choice(range(X.shape[0]))]
        centroids[i] = centroid
    return centroids

def main():
    # 1、get Robot Execution Failures lp1-5 data
    data_name = ['Robot Execution Failures lp1', 'Robot Execution Failures lp2', 'Robot Execution Failures lp3','Robot Execution Failures lp4','Robot Execution Failures lp5']
    for file in data_name:
        filename = file + '.h5'
        f = h5py.File(filename,'r')
        X = f['train_x'][:]
        y = f['train_y'][:] # 1,2,3... np.array
        k = len(np.unique(y))
        centroids = init_random_medoids(X,k)
        cost_time,y_pred,randindex,purity,nmi= kmeans_DTWi(X,y,centroids)
        print("kmeans_DTWi在", file,
              "数据集上聚类的时间为%.2f秒,RI值为%.2f,purity值为%.2f,nmi值为%.2f" % (cost_time, randindex, purity, nmi))
        cost_time, y_pred, randindex, purity, nmi = kmeans_DTWd(X, y, centroids)
        print("kmeans_DTWd在", file,
              "数据集上聚类的时间为%.2f秒,RI值为%.2f,purity值为%.2f,nmi值为%.2f" % (cost_time, randindex, purity, nmi))
        cost_time, y_pred, randindex, purity, nmi = kmeans_aDTWi(X, y, centroids)
        print("kmeans_aDTWi在", file,
              "数据集上聚类的时间为%.2f秒,RI值为%.2f,purity值为%.2f,nmi值为%.2f" % (cost_time, randindex, purity, nmi))
        cost_time, y_pred, randindex, purity, nmi = kmeans_dot(X, y, centroids)
        print("kmeans_dot在", file,
              "数据集上聚类的时间为%.2f秒,RI值为%.2f,purity值为%.2f,nmi值为%.2f" % (cost_time, randindex, purity, nmi))
    # 2、

if __name__ == '__main__':
    main()
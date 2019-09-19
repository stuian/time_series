from sklearn import datasets
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from itertools import cycle, islice

# loaddata
def genTwoCircles(n_samples=1000):
    X,y = datasets.make_circles(n_samples, factor=0.5, noise=0.05)
    return X,y

def euclidDistance(x1, x2, sqrt_flag=False):
    res = np.sum((x1-x2)**2)
    if sqrt_flag:
        res = np.sqrt(res)
    return res

def calEuclidDistanceMatrix(X): # 相似性矩阵、权重矩阵
    X = np.array(X)
    S = np.zeros((len(X), len(X)))
    for i in range(len(X)):
        for j in range(i+1, len(X)):
            S[i][j] = 1.0 * euclidDistance(X[i], X[j])
            S[j][i] = S[i][j]
    return S

def myKNN(S, k, sigma=1.0):
    # S:相似性矩阵
    N = len(S)
    A = np.zeros((N,N))

    for i in range(N):
        dist_with_index = zip(S[i], range(N))
        dist_with_index = sorted(dist_with_index, key=lambda x:x[0])
        neighbours_id = [dist_with_index[m][1] for m in range(k+1)] # xi's k nearest neighbours

        for j in neighbours_id: # xj is xi's neighbour
            A[i][j] = np.exp(-S[i][j]/2/sigma/sigma)
            A[j][i] = A[i][j] # mutually
    return A

def calLaplacianMatrix(adjacentMatrix):

    # compute the Degree Matrix: D=sum(A)
    degreeMatrix = np.sum(adjacentMatrix, axis=1)

    # print degreeMatrix

    # compute the Laplacian Matrix: L=D-A
    laplacianMatrix = np.diag(degreeMatrix) - adjacentMatrix

    # print laplacianMatrix

    # normailze
    # D^(-1/2) L D^(-1/2)
    sqrtDegreeMatrix = np.diag(1.0 / (degreeMatrix ** (0.5)))
    return np.dot(np.dot(sqrtDegreeMatrix, laplacianMatrix), sqrtDegreeMatrix)

def plot(X, y_sp, y_km):
    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                                '#f781bf', '#a65628', '#984ea3',
                                                '#999999', '#e41a1c', '#dede00']),
                                        int(max(y_km) + 1))))
    plt.subplot(121)
    plt.scatter(X[:,0], X[:,1], s=10, color=colors[y_sp])
    plt.title("Spectral Clustering")
    plt.subplot(122)
    plt.scatter(X[:,0], X[:,1], s=10, color=colors[y_km])
    plt.title("Kmeans Clustering")
    plt.show()
    plt.savefig("spectral_clustering.png")

def main():
    data,label = genTwoCircles(n_samples=500) # data是二维的
    Similarity = calEuclidDistanceMatrix(data)
    Adjacent = myKNN(Similarity, k=10)
    print(Adjacent)
    Laplacian = calLaplacianMatrix(Adjacent) # 500x500
    x, V = np.linalg.eig(Laplacian) # 特征值和特征向量
    x = zip(x, range(len(x)))
    x = sorted(x, key=lambda x: x[0],reverse=True)
    # print(len(x))
    H = np.vstack([V[:, i] for (v, i) in x[:500]]).T #v是特征值
    # 行是样本数，列是特征变量
    # print(H.shape)

    sp_kmeans = KMeans(n_clusters=2).fit(H)
    pure_kmeans = KMeans(n_clusters=2).fit(data)

    plot(data, sp_kmeans.labels_, pure_kmeans.labels_)

if __name__ == '__main__':
    main()
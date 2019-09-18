import numpy as np
import h5py

def inner_product(a,b):
    product = np.dot(a,b)
    product = product / (np.sqrt(np.dot(a,a)) * np.sqrt(np.dot(b,b)))
    return 1-product # 越小越好

def multi_similarity(x,y):
    value = 0
    for i in range(x.shape[1]):
        value = value + inner_product(x[:,i],y[:,i])
    return value

def distance_matrix(X):
    n = X.shape[0]
    dist_matrix = np.zeros((n,n))
    for i in range(1,n):
        for j in range(i):
            dist_matrix[i][j] = multi_similarity(X[i],X[j])
            dist_matrix[j][i] = dist_matrix[i][j]
    return dist_matrix

def main():
    file = h5py.File('E:\\Jade\\time_series\\190808_MTS-clustering\\cricket_data.h5', 'r')
    X = file['train_x'][:]
    dist_matrix = distance_matrix(X)
    np.save("data/distance.npy",dist_matrix)

if __name__ == '__main__':
    main()
    # dist_matrix = np.load("data/distance.npy")
    # print(dist_matrix.shape)
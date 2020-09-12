import numpy as np
import heapq
import h5py

# step1 distance between two multivariate series
# distance_matrix.multi_similarity

# step2 local density of every point with RNN
def KNN(n,x,k,D):
    """
    :param X:train_data
    :param x:第x个点
    :return:knn列表
    """
    curr_colum = []
    for i in range(n):
        if i == x:
            curr_colum.append(float('inf'))
        else:
            curr_colum.append(D[x][i])
    temp = map(curr_colum.index, heapq.nsmallest(k, curr_colum))
    return list(temp) # 返回index

def RNN(n,x,k,D):
    count = 0
    for y in range(n):
        if y != x:
            if x in KNN(n,y,k,D):
                count += 1
    return count

def findHigherDensityItems(i,density):
    high_list = []
    for j in range(len(density)):
        if i !=j and density[i] < density[j]:
            high_list.append(j)
    return high_list

def NNDist(i,high_list,D):
    min_dist = float('inf')
    for j in high_list[i]:
        if D[i][j] < min_dist:
            min_dist = D[i][j]
    return min_dist

def distance_to_HDP(D,density):
    """
    distance to higher density points
    :param D: all-pair distance matrix
    :param density: local density vector for all n points in the dataset
    :return: distance;NN(nearest neighbor) distance array of higher density points
    """
    n = D.shape[0]
    # 1、find higher density points list
    high_list = []
    for i in range(n):
        temp_list = findHigherDensityItems(i,density)
        high_list.append(temp_list)
    # 2、sort the list in descending order
    sortIndex = sorted(enumerate(density), key=lambda x: x[1], reverse=True) # [(3, 7), (1, 4),..]
    # 3、calulate the distance to HDP
    distance = np.zeros(n)
    for j in range(1,n):
        distance[sortIndex[j][0]] = NNDist(sortIndex[j][0],high_list,D)
    # the highest density point
    distance[sortIndex[0][0]] = max(distance)
    return distance

# 3、
def densityPeakRNN(k,K,single_distance_between):
    # X:样本数量,样本长度,变量数目;D变成了单变量距离矩阵
    # return centerlabel
    n = single_distance_between.shape[0]
    R = single_distance_between.shape[2]
    D = np.zeros((n,n))
    for i in range(1,n):
        for j in range(i):
            for r in range(R):
                D[i,j] = D[i,j] + (1/R) * np.power(single_distance_between[i,j,r],2)
            D[i,j] = np.sqrt(D[i,j])
            D[j,i] = D[i,j]
    del single_distance_between
    density = np.zeros(n) # 求得每个样本的反近邻数量
    for i in range(n):
        density[i] = RNN(n,i,k,D)
    distance = distance_to_HDP(D,density)
    mean_density = np.mean(density)
    for i in range(len(density)):
        distance[i] = distance[i]*density[i]
    sort_points = sorted(enumerate(distance), key=lambda x: x[1], reverse=True)
    count = 0
    i = 0
    center_points = []
    while count < K:
        # if density[sort_points[i][0]] > mean_density:
        center_points.append(sort_points[i][0])
        count += 1
        i += 1
    return center_points#,density


if __name__ == '__main__':
    file = h5py.File('E:\\Jade\\time_series\\190808_MTS-clustering\\cricket_data.h5', 'r')
    X = file['train_x'][:]
    y = file['train_y'][:]
    D = np.load("pictures/distance.npy")
    # print(len(np.unique(y))) # 12
    k = 12  # knn与峰值k同变量
    center_points = densityPeakRNN(k,X,D)
    print(center_points)
    np.save("pictures/center_points.npy",center_points)

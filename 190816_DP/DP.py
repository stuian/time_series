import numpy as np

"""
DP算法认为聚类中心周围都是比其local_density低的点，且距离那些比其local density高的任意一点相对要远
"""

# 1、
def point_local_density(D,d):
    """
    :param D:all-pair distance matrix
    :param d: cutoff distance
    :return:density;local density list for all n points in the dataset
    """
    n = D.shape[0] # the dataset has n samples
    density = []
    for i in range(n):
        count = 0
        for j in range(n):
            if D[i,j] < d:
                count += 1
        density[i] = count
    return density

# 2、
def findHigherDensityItems(i,density):
    high_list = []
    for j in range(len(density)):
        if i !=j and density[i] < density[j]:
            high_list.append(j)
    return high_list

def NNDist(i,high_list,D):
    mindist = float('inf')
    for j in len(high_list[i]):
        if D[i][j] < mindist:
            mindist = D[i][j]
    return mindist

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
def cluster_center(D,d,k):
    density = point_local_density(D,d)
    distance = distance_to_HDP(D,density)
    for i in range(len(density)):
        distance[i] = distance[i]*density[i]
    sort_points = sorted(enumerate(distance),key=lambda x: x[1],reverse=True)
    center_points = sort_points[:k] # k cluster
    center_points = [center_points[i][0] for i in range(k)] # return point indexs
    return center_points

# 4、
def cluster_assignment(X,n,distance,):
    C = np.zeros(n)
    for i in range(len(X)):
        C[X[i]] = i+1 # 1,2,3,..,k
    for i in range(n):
        if C[sortIndex[i]] == 0: # no cluster yet
            C[sortIndex[i]] = C(NN(sortIndex[i])) # 这部分代码并不完整


def main():
    X = cluster_center(D, d, k)

if __name__ == '__main__':
    main()






import numpy as np

L =np.array([[1.0, 4.2 ,1],
    [1.3, 4.0 , 1],
    [1.0, 4.0 , 1],
    [1.5, 4.3 , 1],
    [2.0, 4.0 , 0],
    [2.3, 3.7 , 0],
    [4.0, 1.0 , 0]])

U =np.array([[1.4, 5.0],
    [1.3, 5.4],
    [2.0, 5.0],
    [4.0, 2.0],
    [5.0, 1.0],
    [5.0, 2.0]])

def distEclud(vecA, vecB):
    '''
    输入：向量A和B
    输出：A和B间的欧式距离
    '''
    return np.sqrt(sum(np.power(vecA - vecB, 2)))


def newCent(L):
    '''
    输入：有标签数据集L
    输出：根据L确定初始聚类中心
    '''
    centroids = []
    label_list = np.unique(L[:, -1])
    for i in label_list:
        L_i = L[(L[:, -1]) == i]
        cent_i = np.mean(L_i, 0)
        centroids.append(cent_i[:-1])
    return np.array(centroids)

    # 随机选取num_clust个样本
    # data = list(data)
    # temp = random.sample(data, num_clust)
    # for ind, i in enumerate(temp):
    #     centroids[ind] = i


def semi_kMeans(L, U, distMeas=distEclud, initial_centriod=newCent):
    '''
    输入：有标签数据集L（最后一列为类别标签）、无标签数据集U（无类别标签）
    输出：聚类结果
    '''
    dataSet = np.vstack((L[:, :-1], U))  # 合并L和U
    label_list = np.unique(L[:, -1])
    k = len(label_list)  # L中类别个数
    m = np.shape(dataSet)[0]

    clusterAssment = np.zeros(m)  # 初始化样本的分配
    centroids = initial_centriod(L)  # 确定初始聚类中心
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):  # 将每个样本分配给最近的聚类中心
            minDist = np.inf
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i] != minIndex: clusterChanged = True #一直到收敛，不在更新，则迭代完成
            clusterAssment[i] = minIndex
    return clusterAssment #聚类结果

clusterResult = semi_kMeans(L,U)
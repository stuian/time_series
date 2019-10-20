import numpy as np
from cluster import multi_similarity

def update_peak(in_cluster,center_label,X,W,D,density):
    K = len(in_cluster)
    for k in range(K):
        length_ck = len(in_cluster[k])
        if length_ck > 1:
            density_k = np.zeros(length_ck)
            distance_w = np.zeros((length_ck,length_ck))
            for i in range(1,length_ck):
                for j in range(i):
                    distance_w[i,j] = multi_similarity(X[in_cluster[k][i]],X[in_cluster[k][j]],W[k])
                    distance_w[j,i] = distance_w[i,j]

            sum_distance = np.sum(distance_w,axis=1)
            min_dist = float('inf')
            minSumLabel = -1
            for i in range(len(sum_distance)):
                if sum_distance[i] < min_dist:
                    minSumLabel = i
                    min_dist = sum_distance[i]

            for i in range(length_ck):
                density_k[i] = density[in_cluster[k][i]]
            max_density = float('-inf')
            maxDensityLabel = -1
            for i in range(length_ck):
                if density_k[i] > max_density:
                    maxDensityLabel = i
                    max_density = density_k[i]

            # 局部密度最大点〖 𝐷𝑀𝑎𝑥〗_𝑘 的 k 近邻中距离点 〖𝑆𝑀𝑖𝑛〗_𝑘 最小的点
            if minSumLabel == maxDensityLabel:
                center_label[k] = in_cluster[k][maxDensityLabel]
            else:
                m = 5
                curr_colum = distance_w[maxDensityLabel,:]
                sorted_distance = sorted(enumerate(curr_colum), key=lambda x: x[1])
                min_dist = float('inf')
                index = -1
                for i in range(len(sorted_distance)):
                    if distance_w[minSumLabel,sorted_distance[i][0]] < min_dist:
                        index = sorted_distance[i][0]
                        min_dist = distance_w[minSumLabel,sorted_distance[i][0]]
                center_label[k] = in_cluster[k][index]
        elif length_ck == 1:
            center_label[k] = in_cluster[k][0]
    return center_label






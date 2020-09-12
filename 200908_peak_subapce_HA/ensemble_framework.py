import numpy as np
import random
from sklearn.cluster import AffinityPropagation
from densityPeakRNN import densityPeakRNN
from update_W_HA import update_subspace
from update_RNNpeak import update_peak
import os
import h5py
from evaluation import RandIndex
from evaluation import Purity
from evaluation import NMI
import scipy.io as io
from save_to_excel import write_excel_xlsx

def kmeans(single_distance_between,W,center_label):
    N = single_distance_between.shape[0]
    R = single_distance_between.shape[2]
    part = np.zeros(N)
    for n in range(N):
        minDist = float('inf')
        for k in range(len(center_label)):
            if n == center_label[k]:
                index = k
                break
            else:
                temp = 0
                for r in range(R):
                    temp += np.power(single_distance_between[n, center_label[k], r], 2) * W[k, r]
                temp = np.sqrt(temp)
                if temp < minDist:
                    minDist = temp
                    index = k
        part[n] = index
    return part

def kmeans_auxi(single_distance_between,W,center_label,auxi_info):
    # auto_info:consensus_matrix
    N = single_distance_between.shape[0]
    R = single_distance_between.shape[2]
    part = np.zeros(N)
    for n in range(N):
        minDist = float('inf')
        for k in range(len(center_label)):
            if n == center_label[k] or auxi_info[n,center_label[k]]==1:
                index = k
                break
            else:
                temp = 0
                for r in range(R):
                    temp += np.power(single_distance_between[n, center_label[k], r], 2) * W[k, r]
                temp = np.sqrt(temp)
                if temp < minDist:
                    minDist = temp
                    index = k
        part[n] = index
    return part

def neigbor_matrix(part):
    N = len(part)
    neigborhood = np.zeros((N,N))
    for i in range(N):
        for j in range(i+1,N):
            if part[i] == part[j]:
                neigborhood[i,j] = 1
            else:
                neigborhood[i,j] = 0
            neigborhood[j,i] = neigborhood[i,j]
    return neigborhood

def mtx_similar2(arr1,arr2):
    '''
    计算对矩阵1的相似度。相减之后对元素取平方再求和。因为如果越相似那么为0的会越多。
    如果矩阵大小不一样会在左上角对齐，截取二者最小的相交范围。
    :param arr1:矩阵1
    :param arr2:矩阵2
    :return:相似度（0~1之间）
    '''
    if arr1.shape != arr2.shape:
        minx = min(arr1.shape[0],arr2.shape[0])
        miny = min(arr1.shape[1],arr2.shape[1])
        differ = arr1[:minx,:miny] - arr2[:minx,:miny]
    else:
        differ = arr1 - arr2
    numera = np.sum(differ**2)
    denom = np.sum(arr1**2)
    similar = 1 - (numera / denom)
    return similar

def find_martrix_min_value(data_matrix):
    '''
    功能：找到矩阵最小值
    '''
    new_data=[]
    for i in range(len(data_matrix)):
        new_data.append(min(data_matrix[i]))
    return min(new_data)

def select_suspace(B,neigborhoods):
    # 计算B个子空间聚类结果的相似性矩阵，反应其子空间的相似性
    relation_matrix = np.zeros((B, B))
    for i in range(B):
        for j in range(i + 1, B):
            relation_matrix[i, j] = mtx_similar2(neigborhoods[i], neigborhoods[j])
            relation_matrix[j, i] = relation_matrix[i, j]

    # 用AP算法筛选子空间
    # 注意：AP里面的相似矩阵是越相似值越大
    center = center_value(relation_matrix)
    # min_value = find_martrix_min_value()
    af = AffinityPropagation(affinity='precomputed', damping=0.5, preference=center).fit(relation_matrix)

    similar_subspace = af.cluster_centers_indices_
    # n_clusters_ = len(cluster_centers_indices)
    labels = af.labels_

    # print('Estimated number of clusters: %d' % n_clusters_)
    # print(cluster_centers_indices)
    # print(labels)
    maxnum_value = np.argmax(np.bincount(labels))  # 簇中元素最多的类别
    various_subspace = []
    for i in range(len(labels)):  # 从B个子空间中筛选
        if labels[i] == maxnum_value:
            various_subspace.append(i)
    return various_subspace,similar_subspace

def center_value(matrix):
    x = []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            x.append(matrix[i,j])
    length = len(x)
    x.sort()
    if length % 2 == 1:
        z = length // 2
        y = x[z]
    else:
        y = (x[length//2]+x[length//2-1])/2
    return y

def kernel(C,S,R,single_distance_between):
    distance = 0
    for i in C:
        for j in S:
            temp = 0
            for r in range(R):
                temp += np.power(single_distance_between[i, j, r], 2)
            temp = np.sqrt(temp)
            distance += temp
    return distance

def subspace_evaluation(part,K,single_distance_between):
    N = len(part)
    R = single_distance_between.shape[2]
    in_cluster = [[] for _ in range(K)]
    for n in range(N):
        in_cluster[int(part[n])].append(n)
    # 获得out_cluster
    out_cluster = []
    for i in range(K):
        temp = []
        for j in range(K):
            if j != i:
                temp = temp + in_cluster[j]
        out_cluster.append(temp)

    S = np.arange(N)
    cost = 0

    for k in range(K):
        M = kernel(in_cluster[k],out_cluster[k],R,single_distance_between)
        D = kernel(in_cluster[k],S,R,single_distance_between)
        if D == 0:
            temp = 0
        else:
            temp = M / D
        cost += temp
    return cost

def subspace_evaluation2(in_cluster,center_label,single_distance_between,W):
    K = len(center_label)
    R = single_distance_between.shape[2]

    pkr = 0
    for k in range(K):
        c = int(center_label[k])
        for n in in_cluster[k]:
            temp = 0
            for r in range(R):
                temp += np.power(single_distance_between[c, n, r],2)*W[k,r]
            temp = np.sqrt(temp)
            pkr += temp
    return pkr

def main():
    choice = 3
    path = './data/'
    if choice == 1:
        data_name = ['lp1','lp2','lp3','lp4']#['cricket','net534', 'uWaveGesture', 'ArticularyWord']
                    # 'lp5','cricket_data','Libras Movement', 'uWaveGesture', 'ArticularyWord','EEG','BCI'
    elif choice == 2:
        data_name = ['pen', 'uWaveGestureLibrary']
    elif choice == 3:
        data_name = ['ECG','ArabicDigits','AUSLAN',"CMUsubject16"] #'wafer298', 'net534', 'vowels', 'ArticulographData', 'char300'
    for file in data_name:
        if choice == 1:
            filename = file + '.h5'
            filename = os.path.join(path, filename)
            f = h5py.File(filename, 'r')
            X = f['train_x'][:]
            print(X.shape)
            y = f['train_y'][:]  # 1,2,3... np.array
        elif choice == 3:
            label_name = file + "_labels.npy"
            y = np.load(os.path.join("./result", label_name))

        print("%s数据集进行子空间集成聚类..." % file)
        # print('数据集大小', X.shape)

        # filename = file + '_subspace'
        eval_file = file + "_ensemble"

        # 获取正常（没有变量权值）的距离矩阵
        PATH = './result/' + file + '_dist.npy'
        single_distance_between = np.load(PATH)

        N = single_distance_between.shape[0]
        R = single_distance_between.shape[2]
        K = len(np.unique(y))

        results = []
        all_evals = []
        all_evals.append(['RI', 'Purity', 'NMI'])
        for step in range(10): # 10
            print("iter %d:" % step)
            B = 30
            # 初始聚类中心label
            center_labels = []

            # 第一步
            neigborhoods = np.zeros((B,N,N))
            parts = np.zeros((B,N))
            W = np.zeros((B, K, R))

            for b in range(B):
                k = random.randint(1, 10)
                center_label = densityPeakRNN(k, K, single_distance_between)  # return center_label
                center_labels.append(center_label)

                W[b] = np.random.random((K, R))
                s = W[b].sum(axis=1)
                for i in range(K):
                    W[b][i] = W[b][i] / s[i]
                cost = float("inf")
                count = 1
                for iter in range(15):
                    part = kmeans(single_distance_between,W[b],center_labels[b])
                    in_cluster = [[] for _ in range(K)]
                    for n in range(N):
                        in_cluster[int(part[n])].append(n)
                    # 获得out_cluster
                    out_cluster = []
                    for i in range(K):
                        temp = []
                        for j in range(K):
                            if j != i:
                                temp = temp + in_cluster[j]
                        out_cluster.append(temp)

                    temp = subspace_evaluation2(in_cluster, center_labels[b], single_distance_between, W[b])

                    if temp == cost :
                        count += 1
                    else:
                        count = 1
                        cost = temp
                    if count == 3:
                        break

                    # 更新子空间
                    W[b] = update_subspace(in_cluster, out_cluster,
                                                              center_labels[b],W[b], single_distance_between)
                    # 更新聚类中心
                    center_labels[b] = update_peak(k,in_cluster, center_labels[b],W[b], single_distance_between)

                neigborhoods[b] = neigbor_matrix(part)
                parts[b] = part
                print("%d subspace has finished!" % b)

            concensus_matrix = np.zeros((N, N))
            for b in range(B):
                concensus_matrix += neigborhoods[b]
            concensus_matrix = concensus_matrix / B
            center = center_value(concensus_matrix)
            af = AffinityPropagation(affinity='precomputed', damping=0.5, preference=np.mean(concensus_matrix)).fit(concensus_matrix)
            labels = af.labels_

            temp = []
            RI_value = RandIndex(labels, y)
            purity_value = Purity(labels, y)
            NMI_value = NMI(labels, y)
            temp.append(RI_value)
            temp.append(purity_value)
            temp.append(NMI_value)
            all_evals.append(temp)
            print(temp)


        # save evaluation
        book_name_xlsx = './result/' + file + '_evaluation.xlsx'
        sheet_name_xlsx = eval_file
        write_excel_xlsx(book_name_xlsx, sheet_name_xlsx, all_evals)


if __name__ == '__main__':
    main()
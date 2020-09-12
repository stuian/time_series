import numpy as np
import os
import h5py
from densityPeakRNN import densityPeakRNN
from save_to_excel import write_excel_xlsx
from evaluation import RandIndex
from evaluation import Purity
from evaluation import NMI
from peak_fullspace import subspace_evaluation
import time

def euclidian(a,b):
    return np.sum(np.square(a-b))

def SGB(x,y):
    # 越小越相似
    y_ = [i for i in reversed(y)]
    CC = np.convolve(x,y_)
    s = np.linalg.norm(x) * np.linalg.norm(y)
    if s == 0 :
        value = 0
    else:
        NCC = CC / s
        index = np.argmax(NCC)
        value = NCC[index]
    dist = 1 - value
    # shift = index - len(x) + 1
    # if shift >= 0:
    #     y = np.pad(y, (shift, 0), 'constant')
    #     y = y[:len(x)]
    # else:
    #     y = np.concatenate((y[-shift:],np.zeros(-shift)))
    return dist #,y

def series_to_centers(single_distance_between, x, center_label, W):
    minDist = float('-inf')
    index = -1
    R = W.shape[1]
    for i in range(len(center_label)):
        if x == int(center_label[i]):
            return i
        else:
            temp = 0
            for r in range(R):
                temp += single_distance_between[x,int(center_label[i]),r] * W[i,r] # 越相似值越大
            temp = np.exp(-0.5*temp)
            if temp > minDist:
                minDist = temp
                index = i
    return index

def kmeans_evaluation(in_cluster,centroids,W,X):
    K = len(centroids)
    R = centroids.shape[2]

    pkr = 0
    for k in range(K):
        c = centroids[k]
        temp = 0
        for n in in_cluster[k]:
            for r in range(R):
                temp +=SGB(c[:,r],X[n,:,r])*W[k,r]
        pkr += temp
    return pkr

def kmeans(X,K):
    N = X.shape[0]
    R = X.shape[2]
    # init center
    count = 0
    center_label = []
    while count != K:
        temp = np.random.randint(0,N)  # 前闭后开
        if temp not in center_label:
            count += 1
            center_label.append(temp)
    centriods = X[center_label]

    iters = 20
    for step in range(iters):
        part = np.zeros(N)
        for n in range(N):
            minDist = float('inf')
            index = -1
            for k in range(K):
                dist = 0
                for r in range(R):
                    dist += euclidian(centriods[k,:,r],X[n,:,r])
                if dist < minDist:
                    minDist = dist
                    index = k
            part[n] = index

        in_cluster = [[] for _ in range(K)]
        for n in range(N):
            ck = part[n]
            in_cluster[int(ck)].append(n)

        if step == 0:
            cost = kmeans_evaluation(in_cluster,centriods,X)
        else:
            temp = kmeans_evaluation(in_cluster,centriods,X)
            if temp == cost:
                print(temp)
                break
            else:
                cost = temp
        print(cost)

        # update centroids
        L = centriods.shape[1]
        for k in range(K):
            temp = np.zeros((L,R))
            for n in in_cluster[k]:
                temp += X[n]
            centriods[k] = temp / len(in_cluster[k])

    return centriods

def init_centers(X,K):
    N = X.shape[0]
    R = X.shape[2]
    Xmax = []
    Xmin = []
    for r in range(R):
        temp = X[:,:,r]
        Xmax.append(temp.max(axis = 0))
        Xmin.append(temp.min(axis = 0))
    Xmax = np.array(Xmax)
    Xmax = Xmax.T
    Xmin = np.array(Xmin)
    Xmin = Xmin.T
    Dmax_min = 0
    for r in range(R):
        Dmax_min += euclidian(Xmax[:,r],Xmin[:,r])
    centriods = kmeans(X,K)
    center_label = []

    priority = np.zeros(N)
    for n in range(N):
        mindist = float('inf')
        for k in range(K):
            temp = 0
            for r in range(R):
                temp += euclidian(centriods[k,:,r],X[n,:,r])
            if temp < mindist:
                mindist = temp
        priority[n] = 1 - (mindist / Dmax_min)

    center_label.append(np.argmax(priority))
    counter = 1

    while counter < K:
        pry = np.zeros(N)
        for n in range(N):
            if n not in center_label:
                mindist = float('inf')
                for i in range(len(center_label)):
                    temp = 0
                    for r in range(R):
                        temp += euclidian(X[center_label[i],:,r],X[n,:,r])
                    if temp < mindist:
                        mindist = temp
                pry[n] = (mindist / Dmax_min) + priority[n]

        max_value = float('-inf')
        index = -1
        for n in range(N):
            if n not in center_label:
                if pry[n] > max_value:
                    max_value = pry[n]
                    index = n
        center_label.append(index)
        counter += 1
    return center_label

def update_subspace(in_cluster,out_cluster,centers,X):
    K = len(in_cluster)
    R = X.shape[2]

    HA = np.zeros((K, R))
    F = np.zeros((K, R))
    M = np.zeros((K, R))

    for k in range(K):
        length_ck = len(in_cluster[k])
        length_nk = len(out_cluster[k])
        if length_ck > 1:
            for r in range(R):
                temp_data = X[:,:,r]
                mean1 = np.sum(temp_data[in_cluster[k]],axis = 0) / length_ck
                mean2 = np.sum(temp_data[out_cluster[k]],axis = 0) / length_nk
                std1 = 0
                for i in range(length_ck):
                    std1 += np.sum(np.square(temp_data[in_cluster[k][i]] - mean1))
                std1 = std1 / length_ck
                std2 = 0
                for j in range(length_nk):
                    std2 += np.sum(np.square(temp_data[out_cluster[k][j]] - mean2))
                std2 = std2 / length_nk

                std1 = np.sqrt(std1)
                std2 = np.sqrt(std2)

                F[k, r] = np.sqrt(1-np.sqrt(2*std1*std2 / (std1**2 + std2**2))*np.exp((-1/4)*(np.sum(np.square(mean1 - mean2)))/(std1**2 + std2**2)))

                for i in in_cluster[k]:
                    M[k,r] = M[k,r] + np.exp(-0.5*euclidian(centers[k][:,r], X[i][:,r]))
                M[k,r] = M[k,r] / length_ck

                HA[k, r] = F[k, r] * M[k, r]
            HA[np.isnan(HA)] = 1 / R
            sumkr = np.sum(HA[k, :])
            if sumkr != 0:
                HA[k, :] = HA[k, :] / sumkr
            else:
                HA[k, :] = 1 / R
        else:
            HA[k, :] = 1 / R
    return HA

def update_peak(in_cluster,centers,X):
    new_centers = np.zeros(np.shape(centers))
    K = len(new_centers)

    for k in range(K):
        for i in in_cluster[k]:
            new_centers[k] += X[i]
        new_centers[k] = new_centers[k] / len(in_cluster[k])
    return new_centers

def weighted_euclidian(a,b,w):
    return np.exp(-0.5*np.sum(np.square(a-b))*w)

def  Attribute_WeightedOCIL(y,center_label,single_distance_between):
    K = len(np.unique(y))
    print('聚类簇数：', K)
    N = single_distance_between.shape[0]
    R = single_distance_between.shape[2]

    # init centers
    centriods = X[center_label]
    W = np.ones((K, R))
    W = W / R

    iter = 15
    all_evals = []
    all_evals.append(['RI', 'Purity', 'NMI','cost'])

    for step in range(1, iter + 1):
        part = np.zeros(N)
        in_cluster = [[] for _ in range(K)]
        for n in range(N):
            temp = []
            for k in range(K):
                dist = 0
                for r in range(R):
                    dist += SGB(centriods[k,:,r], X[n,:,r])* W[k,r]
                temp.append(dist)
            # print(temp)
            # temp = temp / np.sum(temp)
            index = np.argmin(temp)
            part[n] = index
            in_cluster[index].append(n)

        for k in range(K):
            if in_cluster[k] == []:
                temp = np.random.randint(0, N)
                in_cluster[k].append(temp)
                part[temp] = k

        out_cluster = []
        for i in range(K):
            temp = []
            for j in range(K):
                if j != i:
                    temp = temp + in_cluster[j]
            out_cluster.append(temp)

        # evaluation
        temp = []
        RI_value = round(RandIndex(part, y),4)
        purity_value = round(Purity(part, y),4)
        NMI_value = round(NMI(part, y),4)
        cost = kmeans_evaluation(in_cluster,centriods,W,X)
        print(RI_value, purity_value,NMI_value,cost)
        temp.append(RI_value)
        temp.append(purity_value)
        temp.append(NMI_value)
        temp.append(cost)
        all_evals.append(temp)

        W = update_subspace(in_cluster, out_cluster, centriods, X)

        L = centriods.shape[1]
        for k in range(K):
            temp = np.zeros((L,R))
            for n in in_cluster[k]:
                temp += X[n]
            centriods[k] = temp / len(in_cluster[k])

    return all_evals


def main():
    path = './data/'
    data_name = ["cricket", 'ArticularyWord']
    for file in data_name:
        print(file)
        filename = file + '.h5'
        filename = os.path.join(path, filename)
        f = h5py.File(filename, 'r')
        X = f['train_x'][:]
        # print(X.shape)
        y = f['train_y'][:]

        # 获取正常（没有变量权值）的距离矩阵
        PATH = './result/' + file + '_dist.npy'
        single_distance_between = np.load(PATH)

        iters = 10
        centername = file + '_centerlabels.npy'
        center_path = os.path.join("./result", centername)
        center_labels = np.load(center_path)

        eval_file = file + "_WOCIL"
        evals = []
        for iter in range(iters):
            center_label = center_labels[iter]
            all_evals = Attribute_WeightedOCIL(y,center_label,single_distance_between)
            evals.append(all_evals[-1])

        # save evaluation
        book_name_xlsx = './result/' + file + '_WOCIL_evaluation.xlsx'
        sheet_name_xlsx = eval_file
        write_excel_xlsx(book_name_xlsx, sheet_name_xlsx, evals)


if __name__ == '__main__':
    main()
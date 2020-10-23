import numpy as np
import random
import os
import h5py
from sklearn.cluster import SpectralClustering
from evaluation import RandIndex
from evaluation import Purity
from evaluation import NMI
import scipy.io as io
from save_to_excel import write_excel_xlsx

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
    return dist

def kmeans_evaluation(in_cluster,centroids,X):
    K = len(centroids)
    R = centroids.shape[2]

    pkr = 0
    for k in range(K):
        c = centroids[k]
        temp = 0
        for n in in_cluster[k]:
            for r in range(R):
                temp += SGB(c[:,r],X[n,:,r])
        pkr += temp
    return pkr

def kmeans(X,K):
    N = X.shape[0]
    # init center
    count = 0
    center_label = []
    while count != K:
        temp = np.random.randint(0,N)  # 前闭后开
        if temp not in center_label:
            count += 1
            center_label.append(temp)
    centriods = X[center_label]

    R = centriods.shape[2]

    part = np.zeros(N)
    for n in range(N):
        minDist = float('inf')
        index = -1
        for k in range(K):
            dist = 0
            for r in range(R):
                dist += SGB(centriods[k, :, r], X[n, :, r])
            if dist < minDist:
                minDist = dist
                index = k
        part[n] = index

    in_cluster = [[] for _ in range(K)]
    for n in range(N):
        ck = part[n]
        in_cluster[int(ck)].append(n)

    for k in range(K):
        if in_cluster[k] == []:
            temp = np.random.randint(0, N)
            in_cluster[k].append(temp)
            part[temp] = k

    # cost = kmeans_evaluation(in_cluster, centriods, X)
    # print(cost)

    iters = 14
    for step in range(iters):
        # update centroids
        L = centriods.shape[1]
        for k in range(K):
            temp = np.zeros((L,R))
            for n in in_cluster[k]:
                temp += X[n]
            centriods[k] = temp / len(in_cluster[k])

        part = np.zeros(N)
        for n in range(N):
            minDist = float('inf')
            index = -1
            for k in range(K):
                dist = 0
                for r in range(R):
                    dist += SGB(centriods[k, :, r], X[n, :, r])
                if dist < minDist:
                    minDist = dist
                    index = k
            part[n] = index

        in_cluster = [[] for _ in range(K)]
        for n in range(N):
            ck = part[n]
            in_cluster[int(ck)].append(n)

        for k in range(K):
            if in_cluster[k] == []:
                temp = np.random.randint(0, N)
                in_cluster[k].append(temp)
                part[temp] = k

        # temp = kmeans_evaluation(in_cluster, centriods, X)
        # if temp == cost:
        #     # print(temp)
        #     break
        # else:
        #     cost = temp
        # print(cost)

    return part,centriods

def FS_RS_NC(X,B,y):
    #K_set:2-30
    N = X.shape[0]
    R = X.shape[2]
    parts = np.zeros((B,N))

    sigmas = []
    for b in range(B):
        print(b)

        samples = []
        for n in range(N):
            samples.append(n)
        variables = []
        for r in range(R):
            variables.append(r)

        part = np.zeros(N)
        K = len(np.unique(y))
        # K = np.random.randint(2,30)
        variable_rate = random.uniform(0.2, 0.7)
        sample_rate = random.uniform(0.2, 0.7)

        random.shuffle(variables)
        selected_variables_num = int(variable_rate * R)
        if selected_variables_num == 0:
            selected_variables_num = 1
        selected_variables = variables[:selected_variables_num]
        # print(selected_variables)
        reduced_X = X[:, :, selected_variables]

        random.shuffle(samples)
        selected_samples_num = int(sample_rate * N)
        if selected_samples_num < K:
            selected_samples_num = K

        selected_samples = samples[:selected_samples_num]
        unselected_samples = samples[selected_samples_num:]


        data = reduced_X[selected_samples]
        # print(data.shape)
        # print(K)

        part_result, centriods = kmeans(data, K)
        for i in range(len(selected_samples)):
            part[selected_samples[i]] = part_result[i]
        for j in unselected_samples:
            minDist = float('inf')
            index = -1
            for k in range(K):
                dist = 0
                for r in range(selected_variables_num):
                    dist += SGB(centriods[k, :, r], reduced_X[j, :, r])
                if dist < minDist:
                    minDist = dist
                    index = k
            part[j] = index
        parts[b] = part

        in_cluster = [[] for _ in range(K)]
        for n in range(N):
            in_cluster[int(part[n])].append(n)

        for k in range(K):
            if in_cluster[k] == []:
                temp = np.random.randint(0, N)
                in_cluster[k].append(temp)
                parts[b][temp] = k

        for k in range(K):
            samples = in_cluster[k]
            cluster_length = len(samples)
            sigma = 0
            for i in samples:
                sigma += Silhouette_Coefficient(i, k, reduced_X, in_cluster, centriods)
            sigmas.append(sigma / cluster_length)

        if b == 0:
            H = np.zeros((N,K))
            for n in range(N):
                H[n,int(part[n])] = 1
        else:
            temp = np.zeros((N,K))
            for n in range(N):
                temp[n,int(part[n])] = 1
            H = np.hstack((H,temp))
    # S = H.dot(H.T) / B
    return H,parts,sigmas

def Silhouette_Coefficient(i,k,X,in_cluster,centriods):
    # 分子的第一项,x与其同簇的样本的距离和
    R = X.shape[2]
    first = 0
    temp = 0
    K = len(in_cluster)
    for n in in_cluster[k]:
        for r in range(R):
            temp += euclidian(X[i, :, r], X[n, :, r])
    first += temp
    first = first / len(in_cluster[k])
    # 求最近邻的簇
    minDist = float('inf')
    index = -1
    for j in range(K):
        if j != k:
            dist = 0
            for r in range(R):
                dist += SGB(centriods[j, :, r], X[i, :, r])
            if dist < minDist:
                minDist = dist
                index = j
    second = 0
    for n in in_cluster[index]:
        for r in range(R):
            second += euclidian(X[i, :, r], X[n, :, r])
    second = second / len(in_cluster[index])
    return abs(first-second)/max(first,second)

def main():
    path = './data/'
    data_name = ["lp1"]#['wafer298', 'net534', 'JapaneseVowels',"CMUsubject16","ArabicDigits"]
     # "uWaveGestureLibrary",'net534','lp5', 'ArticularyWord','cricket_data', 'uWaveGesture', 'BCI','EEG'
    for file in data_name:
        print(file)
        filename = file + '.h5'
        filename = os.path.join(path, filename)
        f = h5py.File(filename, 'r')
        X = f['train_x'][:]
        y = f['train_y'][:]
        N = len(y)
        B = 30
        all_evals = []
        evals = []
        for iter in range(10):
            print("iter:",iter)
            H,parts,sigmas = FS_RS_NC(X, B,y)

            results = []
            for b in range(B):
                # evaluation
                temp = []
                RI_value = RandIndex(parts[b], y)
                purity_value = Purity(parts[b], y)
                NMI_value = NMI(parts[b], y)
                temp.append(RI_value)
                temp.append(purity_value)
                temp.append(NMI_value)
                results.append(temp)
            results = np.array(results)
            temp = []
            temp.append(np.mean(results[:,0]))
            temp.append(np.mean(results[:,1]))
            temp.append(np.mean(results[:,2]))
            evals.append(temp)

            # clustering-level consistency
            # cluster-level consistency
            # Silhouette Coefficient
            # sigmas = []
            # for b in range(B):
            #     part = parts[b]
            #     K = len(np.unique(part))
            #     in_cluster = [[] for _ in range(K)]
            #     for n in range(N):
            #         in_cluster[int(part[n])].append(n)
            #     for k in range(K):
            #         samples = in_cluster[k]
            #         cluster_length = len(samples)
            #         sigma = 0
            #         for i in samples:
            #             sigma += Silhouette_Coefficient(i,k,X,in_cluster,centriods)
            #         sigmas.append(sigma/cluster_length)
            W = np.diag(np.array(sigmas))
            S = np.dot(H.dot(W),H.T) / B
            # 1、spectral cluster
            K = len(np.unique(y))
            pred_y = SpectralClustering(n_clusters=K, assign_labels="discretize", random_state=2020,
                                        affinity='precomputed').fit_predict(S)

            temp = []
            RI_value = RandIndex(y, pred_y)
            purity_value = Purity(y, pred_y)
            NMI_value = NMI(y, pred_y)
            temp.append(RI_value)
            temp.append(purity_value)
            temp.append(NMI_value)
            all_evals.append(temp)
            print("SP:",temp)

            # 2、CSPA

        eval_file1 = file + "_WECR_KM_Single"
        book_name_xlsx = './result/' + file + '_evaluation.xlsx'
        sheet_name_xlsx = eval_file1
        write_excel_xlsx(book_name_xlsx, sheet_name_xlsx, evals)

        eval_file1 = file + "_WECR_KM_SP"
        book_name_xlsx = './result/' + file + '_evaluation.xlsx'
        sheet_name_xlsx = eval_file1
        write_excel_xlsx(book_name_xlsx, sheet_name_xlsx, all_evals)

if __name__ == '__main__':
    main()
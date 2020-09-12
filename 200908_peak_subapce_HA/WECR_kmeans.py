import numpy as np
import random
import os
import h5py

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

        # temp = kmeans_evaluation(in_cluster, centriods, X)
        # if temp == cost:
        #     # print(temp)
        #     break
        # else:
        #     cost = temp
        # print(cost)

    return part,centriods

def FS_RS_NC(X,B):
    #K_set:2-30
    N = X.shape[0]
    R = X.shape[2]
    parts = np.zeros((B,N))
    samples = []
    for n in range(N):
        samples.append(n)
    variables = []
    for r in range(R):
        variables.append(r)
    for b in range(B):
        part = np.zeros(N)
        K = np.random.randint(2,30)
        variable_rate = random.uniform(0.2, 0.7)
        sample_rate = random.uniform(0.2, 0.7)
        random.shuffle(samples)
        selected_samples = samples[:int(sample_rate * N)]
        unselected_samples = samples[int(sample_rate * N):]
        data = X[selected_samples]
        random.shuffle(variables)
        selected_variables = variables[:int(variable_rate * R)]
        data = data[:, :, selected_variables]
        part_result, centriods = kmeans(data, K)
        for i in range(len(selected_samples)):
            part[selected_samples[i]] = part_result[i]
        for j in unselected_samples:
            minDist = float('inf')
            index = -1
            for k in range(K):
                dist = 0
                for r in range(R):
                    dist += SGB(centriods[k, :, r], X[j, :, r])
                if dist < minDist:
                    minDist = dist
                    index = k
            part[j] = index
        parts[b] = part
        if b == 0:
            H = np.zeros((N,K))
            for n in range(N):
                H[n,int(part[n])] = 1
        else:
            temp = np.zeros((N,K))
            for n in range(N):
                temp[n,int(part[n])] = 1
            H = np.hstack(H,temp)
    # S = H.dot(H.T) / B
    return H,parts

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
    data_name = ['lp5']
     # 'Libras Movement','lp5', 'ArticularyWord','cricket_data', 'uWaveGesture', 'BCI','EEG'
    for file in data_name:
        filename = file + '.h5'
        filename = os.path.join(path, filename)
        f = h5py.File(filename, 'r')
        X = f['train_x'][:]
        y = f['train_y'][:]
        N = len(y)
        B = 20
        H,parts = FS_RS_NC(X, B)
        # clustering-level consistency
        # cluster-level consistency
        # Silhouette Coefficient
        sigmas = []
        for b in range(B):
            part = parts[b]
            K = len(np.unique(part))
            in_cluster = [[] for _ in range(K)]
            for n in range(N):
                in_cluster[int(part[n])].append(n)
            for k in range(K):
                samples = in_cluster[k]
                cluster_length = len(samples)
                sigma = 0
                for i in samples:
                    sigma += Silhouette_Coefficient(i,)
                sigmas.append(sigma/cluster_length)
        W = np.diag(np.array(sigmas))
        S = np.dot(H.dot(W),H.T) / B
        # 1、spectral cluster
        # 2、CSPA

if __name__ == '__main__':
    main()
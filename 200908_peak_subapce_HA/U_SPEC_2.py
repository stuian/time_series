import numpy as np
import random
import heapq
import os
import h5py
from sklearn.cluster import SpectralClustering
from evaluation import RandIndex
from evaluation import Purity
from evaluation import NMI
from save_to_excel import write_excel_xlsx
import warnings
warnings.filterwarnings('ignore')

def SBD(x,y):
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

def euclidian(a,b):
    return np.sum(np.square(a-b))

def kmeans_evaluation(in_cluster, center_label, RpFea,single_distance_between):
    K = len(center_label)
    R = single_distance_between.shape[2]

    pkr = 0
    for k in range(K):
        c = center_label[k]
        temp = 0
        for n in in_cluster[k]:
            for r in range(R):
                temp += single_distance_between[RpFea[c],RpFea[n],r]
        pkr += temp
    return pkr

def kmeans(RpFea,K,single_distance_between):
    num = len(RpFea)
    # init center
    count = 0
    center_label = []
    while count != K:
        temp = np.random.randint(0,num)  # 前闭后开
        if temp not in center_label:
            count += 1
            center_label.append(temp)

    R = single_distance_between.shape[2]

    cost = float("inf")
    for iter in range(15):
        part = np.zeros(num)
        for n in range(num):
            minDist = float('inf')
            index = -1
            for k in range(K):
                dist = 0
                for r in range(R):
                    dist += single_distance_between[RpFea[n],RpFea[center_label[k]],r]
                if dist < minDist:
                    minDist = dist
                    index = k
            part[n] = index

        in_cluster = [[] for _ in range(K)]
        for n in range(num):
            ck = part[n]
            in_cluster[int(ck)].append(n)

        for k in range(K):
            if in_cluster[k] == []:
                temp = np.random.randint(0, num)
                in_cluster[k].append(temp)
                part[temp] = k

        temp = kmeans_evaluation(in_cluster, center_label, RpFea,single_distance_between)

        if temp == cost:
            count += 1
        else:
            cost = temp
            count = 1
        if count == 2:
            break

        center_label = update_center(in_cluster, center_label, single_distance_between, RpFea)

    for k in range(K):
        temp = center_label[k]
        center_label[k] = RpFea[temp]

    return part,center_label

def update_center(in_cluster,center_label,single_distance_between,selected_samples):
    K = len(center_label)
    R = single_distance_between.shape[2]

    for k in range(K):
        length_ck = len(in_cluster[k])
        if length_ck > 1:
            distance_w = np.zeros((length_ck, length_ck))
            for i in range(1, length_ck):
                for j in range(i):
                    for r in range(R):
                        distance_w[i, j] = distance_w[i, j] + single_distance_between[selected_samples[in_cluster[k][i]], selected_samples[in_cluster[k][j]], r]
                    distance_w[j, i] = distance_w[i, j]

            # minSsumLabel
            sum_distance = np.sum(distance_w, axis=1)
            min_dist = float('inf')
            minSumLabel = -1
            for i in range(len(sum_distance)):
                if sum_distance[i] < min_dist:
                    minSumLabel = i  # 0-length_ck
                    min_dist = sum_distance[i]

            center_label[k] = in_cluster[k][minSumLabel]
        elif length_ck == 1:
            center_label[k] = in_cluster[k][0]
    return center_label

def Hybrid_Representative_Selection(X,single_distance_between):
    cntTimes = 10
    N = X.shape[0]
    R = X.shape[2]
    # p = 5

    sample_rates = random.uniform(0.2, 0.7)
    p = int(sample_rates*N)
    resamples = p*cntTimes
    if resamples>N:
        resamples = N
    samples = []
    for n in range(N):
        samples.append(n)
    random.shuffle(samples)
    selected_samples = samples[:resamples]
    # data = X[selected_samples]
    # _,RpFea = kmeans(data,p)

    num = len(selected_samples)
    # init center
    count = 0
    center_label = []
    while count != p:
        temp = np.random.randint(0, num)  # 前闭后开
        if temp not in center_label:
            count += 1
            center_label.append(temp)

    cost = float("inf")
    count = 0
    for step in range(15):
        part = np.zeros(num)
        for n in range(num):
            minDist = float('inf')
            index = -1
            for k in range(p):
                dist = 0
                for r in range(R):
                    dist += single_distance_between[selected_samples[center_label[k]],selected_samples[n],r]
                if dist < minDist:
                    minDist = dist
                    index = k
            part[n] = index

        in_cluster = [[] for _ in range(p)]
        for n in range(num):
            ck = part[n]
            in_cluster[int(ck)].append(n)

        for k in range(p):
            if in_cluster[k] == []:
                temp = np.random.randint(0, num)
                in_cluster[k].append(temp)
                part[temp] = k

        temp = kmeans_evaluation(in_cluster, center_label, selected_samples, single_distance_between)

        if temp == cost:
            count += 1
        else:
            cost = temp
            count = 1
        if count == 2:
            break

        # update centers
        center_label = update_center(in_cluster, center_label, single_distance_between,selected_samples)

    for k in range(p):
        temp = center_label[k]
        center_label[k] = selected_samples[temp]

    return center_label,p

def Approxi_of_KNearest_Rp(X,single_distance_between):
    # N x p submatix
    N = X.shape[0]
    R = X.shape[2]
    K = 5
    RpFea,p = Hybrid_Representative_Selection(X,single_distance_between)
    submatrix = np.zeros((N,p))
    z = int(np.sqrt(p))
    part,centers = kmeans(RpFea, z,single_distance_between)
    in_cluster = [[] for _ in range(z)]
    for i in range(len(RpFea)):
        in_cluster[int(part[i])].append(i)
    for n in range(N):
        # find nearest center
        dist = float('inf')
        index = -1
        for i in range(z):
            temp = 0
            for r in range(R):
                temp += single_distance_between[centers[i],n,r]
            if temp < dist:
                dist = temp
                index = i
        # find nearest RpFea in the nearest center
        # dist = float('inf')
        # rl = -1
        # for j in in_cluster[index]:
        #     temp = 0
        #     for r in range(R):
        #         temp += SBD(centers[index, :, r], RpFea[j, :, r])
        #     if temp < dist:
        #         dist = temp
        #         rl = j
        r1 = centers[index] # original label

        # find k1 nearest neigbor of rl
        k1 = 10*K
        curr_colum = []
        for i in range(len(RpFea)):
            temp = 0
            for r in range(R):
                temp += single_distance_between[r1,RpFea[i],r]#SBD(RpFea[rl, :, r],RpFea[i, :, r])
            curr_colum.append(temp)
        k1_nearest = map(curr_colum.index, heapq.nsmallest(k1, curr_colum))
        k1_nearest = list(k1_nearest)[:K]
        sigma = []
        for j in k1_nearest:
            temp = 0
            for r in range(R):
                temp += single_distance_between[n,RpFea[j],r]#SBD(X[n, :, r], RpFea[j, :, r])
            sigma.append(temp)
        mid = np.mean(sigma)
        for i in range(len(k1_nearest)):
            submatrix[n,k1_nearest[i]] = np.exp(-sigma[i]/(2*mid**2))

    zeros1 = np.zeros((p,p))
    zeros2 = np.zeros((N,N))
    temp1 = np.concatenate((zeros1,submatrix.T),axis = 1)
    temp2 = np.concatenate((submatrix,zeros2),axis = 1)
    G = np.concatenate((temp1,temp2),axis=0)

    # D = []
    # for i in range(N):
    #     D.append(np.sum(submatrix[i,:]))
    # D = np.diag(np.array(D))
    # G = np.dot(submatrix.T,np.linalg.inv(D)).dot(submatrix)

    return G

        # D = []
        # for i in range(N+p):
        #     D.append(np.sum(G[i,:]))
        # D = np.diag(np.array(D))
        # L = D - G

        # D = []
        # for i in range(N):
        #     D.append(np.sum(submatrix[i,:]))
        # D = np.diag(np.array(D))
        # D_I = np.linalg.inv(D)
        # E = np.dot(submatrix.T,D_I).dot(submatrix)
        # Dr = []
        # for i in range(p):
        #     Dr.append(np.sum(E[i,:]))
        # Dr = np.diag(np.array(Dr))
        # L = Dr - E

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

def main():
    choice = 1
    path = './data/'
    if choice == 1:
        data_name = ['wafer298', 'net534', 'JapaneseVowels',"CMUsubject16","ArabicDigits"]#['lp4','cricket','Libras','net534', 'ArticularyWord']
        # 'lp5','cricket_data','Libras Movement', 'uWaveGesture', 'ArticularyWord','EEG','BCI'
    elif choice == 2:
        data_name = ['pen', 'uWaveGestureLibrary']
    elif choice == 3:
        data_name = ["ArabicDigits"]  # 'wafer298', 'net534', 'vowels', 'ArticulographData', 'char300'
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

        print("%s数据集进行U_SPEC集成聚类..." % file)

        eval_file = file + "_USPEC"

        # 获取正常（没有变量权值）的距离矩阵
        PATH = './result/' + file + '_dist.npy'
        single_distance_between = np.load(PATH)

        # Nxp
        # nearest K

        B = 30
        all_evals = []
        evals = []
        N = len(y)
        K = len(np.unique(y))
        for iter in range(10):
            print(iter)
            temp_RI_value = 0
            temp_purity_value = 0
            temp_NMI_value = 0
            for b in range(B):
                G = Approxi_of_KNearest_Rp(X,single_distance_between)
                print("G finished!")

                pred_y = SpectralClustering(n_clusters=K, assign_labels="discretize", random_state=2020,
                                            affinity='precomputed').fit_predict(G)
                pred_y = pred_y[:N]

                RI_value = RandIndex(y, pred_y)
                purity_value = Purity(y, pred_y)
                NMI_value = NMI(y, pred_y)

                temp_RI_value += RI_value
                temp_purity_value += purity_value
                temp_NMI_value += NMI_value

                if b == 0:
                    H = np.zeros((N, K))
                    for n in range(N):
                        H[n, int(pred_y[n])] = 1
                else:
                    temp = np.zeros((N, K))
                    for n in range(N):
                        temp[n, int(pred_y[n])] = 1
                    H = np.hstack((H, temp))


            zeros1 = np.zeros((B*K, B*K))
            zeros2 = np.zeros((N, N))
            temp1 = np.concatenate((zeros1, H.T), axis=1)
            temp2 = np.concatenate((H, zeros2), axis=1)
            G = np.concatenate((temp1, temp2), axis=0)

            # D = []
            # for i in range(N):
            #     D.append(np.sum(H[i, :]))
            # D = np.diag(np.array(D))
            # G = np.dot(H.T, np.linalg.inv(D)).dot(H)

            pred_y = SpectralClustering(n_clusters=K, assign_labels="discretize", random_state=2020,
                                        affinity='precomputed').fit_predict(G)
            pred_y = pred_y[:N]

            temp = []
            RI_value = RandIndex(y, pred_y)
            purity_value = Purity(y, pred_y)
            NMI_value = NMI(y, pred_y)
            temp.append(RI_value)
            temp.append(purity_value)
            temp.append(NMI_value)
            all_evals.append(temp)
            print("SP:", temp)

            temp = []
            temp_RI_value /= 30
            temp_purity_value /= 30
            temp_NMI_value /= 30
            temp.append(temp_RI_value)
            temp.append(temp_purity_value)
            temp.append(temp_NMI_value)
            evals.append(temp)

        eval_file1 = file + "_U_SPEC"
        book_name_xlsx = './result/' + file + '_evaluation.xlsx'
        sheet_name_xlsx = eval_file1
        write_excel_xlsx(book_name_xlsx, sheet_name_xlsx, all_evals)

        eval_file1 = file + "_U_SPEC_single"
        book_name_xlsx = './result/' + file + '_evaluation.xlsx'
        sheet_name_xlsx = eval_file1
        write_excel_xlsx(book_name_xlsx, sheet_name_xlsx, evals)


if __name__ == '__main__':
    main()
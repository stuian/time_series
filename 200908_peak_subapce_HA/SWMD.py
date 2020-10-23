import os
import h5py
import numpy as np
from scipy import linalg as LA
from evaluation import RandIndex
from evaluation import Purity
from evaluation import NMI
from save_to_excel import write_excel_xlsx

def VPCA(X,Ps):
    M = X.shape[0]
    L = X.shape[1]
    N = X.shape[2]
    V = np.zeros((N,M,L))
    F = np.zeros((N,M,Ps))
    Y = np.zeros((M,N,Ps))
    for n in range(N):
        V[n] = X[:,:,n]
        variance_matrix = V[n] - np.sum(V[n],axis=0) / M # 减去均值
        covariance_matrix = variance_matrix.T.dot(variance_matrix)
        eigvalues, eigvectors = LA.eig(covariance_matrix)
        indices = np.argsort(eigvalues)[:Ps]
        Un = eigvectors[:, indices]
        Fn = V[n].dot(Un)
        F[n] = Fn.real
    for m in range(M):
        for n in range(N):
            Y[m,n,:] = F[n,m,:]
    return Y

def Spatial_Weighted_Matrix_Dist(X,Y):
    N = X.shape[0]
    Ps = X.shape[1]
    sigma = 1-1/Ps
    distance = 0
    for i in range(N):
        for j in range(Ps):
            # u = i + (j - 1)*N
            for n in range(N):
                for p in range(Ps):
                    # w = n + (p - 1)*N
                    S = (1/(2*np.pi*sigma**2))*np.exp(-np.sqrt((i-n)**2 + (j-p)**2)/(2*sigma**2))
                    d = S*(X[i,j]-Y[i,j])*(X[n,p]-Y[n,p])
                    distance += d
    distance = np.sqrt(distance)
    return distance

def vectorized(a):
    for i in range(a.shape[1]):
        if i == 0:
            vector = a[:, i]
        else:
            temp = a[:, i]
            vector = np.concatenate((vector, temp))
    return vector


def SWMDist(X,Y):
    N = X.shape[0]
    Ps = X.shape[1]
    sigma = 1 - 1 / Ps
    x = vectorized(X)
    Nps = len(x)
    y = vectorized(Y)
    S = np.zeros((Nps,Nps))
    for i in range(N):
        for j in range(Ps):
            u = i + (j - 1)*N
            for n in range(N):
                for p in range(Ps):
                    w = n + (p - 1)*N
                    S[u,w] = (1 / (2 * np.pi * sigma ** 2)) * np.exp(
                        -np.sqrt((i - n) ** 2 + (j - p) ** 2) / (2 * sigma ** 2))
    a = (x-y).reshape(1, Nps)
    b = (x-y).reshape(Nps,1)
    return np.sqrt(np.dot(a.dot(S),b))[0,0]

def main():
    path = './data'
    data_name = ['wafer298','net534','cricket', 'JapaneseVowels', 'CharacterTrajectories', 'CMUsubject', 'ArticularyWord',
                 'Libras', 'ArabicDigits']  # 'lp5','cricket','ArticularyWord','Libras'
    for file in data_name:
        filename = file + '.h5'
        filename = os.path.join(path, filename)
        f = h5py.File(filename, 'r')
        X = f['train_x'][:]
        # print(X.shape)
        y = f['train_y'][:]

        print(file,'数据集进行SWMD聚类')

        K = len(np.unique(y))
        M = X.shape[0]
        L = X.shape[1]
        N = X.shape[2]
        Ps = int(L*0.8)
        b = 8

        Y = VPCA(X,Ps)
        print('dimension reduction finished!')

        evals = []
        for count in range(10):
            print("第%d次运行算法!" % count)
            # init centroids
            count = 0
            center_label = []
            while count != K:
                temp = np.random.randint(0,M)  # 前闭后开
                if temp not in center_label:
                    count += 1
                    center_label.append(temp)
            centroids = Y[center_label]

            iter = 10
            all_evals = []
            all_evals.append(['RI', 'Purity', 'NMI'])  #
            for step in range(1, iter + 1):
                part = np.zeros((M,K))
                for m in range(M):
                    for k in range(K):
                        d = Spatial_Weighted_Matrix_Dist(Y[m],centroids[k])
                        temp = -2/(b-1)
                        if d == 0:
                            part[m, k] = 1
                            for i in range(k):
                                part[m, i] = 0
                            break
                        else:
                            d = np.power(d,temp)
                            part[m,k] = d
                for m in range(M):
                    sumkr = np.sum(part[m, :])
                    part[m, :] = part[m, :] / sumkr

                # partition results
                P = np.zeros(M)
                for m in range(M):
                    indices = np.argmax(part[m,:])
                    P[m] = indices

                # evaluation
                temp = []
                RI_value = RandIndex(P, y)
                purity_value = Purity(P, y)
                NMI_value = NMI(P, y)
                print(RI_value, purity_value,NMI_value)
                temp.append(RI_value)
                temp.append(purity_value)
                temp.append(NMI_value)
                all_evals.append(temp)

                # update fuzzy clluster centers
                centroids = np.zeros((K,N,Ps))
                for k in range(K):
                    for m in range(M):
                        centroids[k] += part[m,k]*Y[m]
                    centroids[k] = centroids[k] / np.sum(part[:,k])

            if count == 1:
                evals.append(all_evals[0])
                evals.append(all_evals[-1])
            else:
                evals.append(all_evals[-1])

        # save evaluation
        eval_file = file + "_SWMD"
        book_name_xlsx = './result/' + file + '_evaluation.xlsx'
        sheet_name_xlsx = eval_file
        write_excel_xlsx(book_name_xlsx, sheet_name_xlsx, evals)


if __name__ == '__main__':
    main()
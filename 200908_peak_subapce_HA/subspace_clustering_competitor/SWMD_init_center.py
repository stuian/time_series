import os
import h5py
import numpy as np
from scipy import linalg as LA
from evaluation import RandIndex
from evaluation import Purity
from evaluation import NMI
from save_to_excel import write_excel_xlsx
from densityPeakRNN import densityPeakRNN
import time
from SWMD import SWMDist

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

def SGB(x,y):
    # 越小越相似
    y_ = [i for i in reversed(y)]
    CC = np.convolve(x,y_)
    # s = np.linalg.norm(x) * np.linalg.norm(y)
    a = np.sqrt(np.sum(np.square(x)))
    b = np.sqrt(np.sum(np.square(y)))
    s = a*b
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

def main():
    path = './data'
    data_name = ['ArabicDigits']#"ArabicDigits","uWaveGestureLibrary"
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
        r = X.shape[2]
        # print(L)
        # Ps = int(0.8*r) # 2,3,4
        # b = 8
        dimensions = [1,2,3,4]

        # init centers
        PATH = './result/' + file + '_dist.npy'
        single_distance_between = np.load(PATH)

        k = 5
        center_label = densityPeakRNN(k, K, single_distance_between)  # return center_label
        print("initial center_label: ", center_label)

        for Ps in dimensions:
            time_begin = time.time()
            print(X.shape)
            temp = X.transpose(0, 2, 1)
            Y = VPCA(temp,Ps)
            Y = Y.transpose(0, 2, 1)
            print(Y.shape)
            print('dimension reduction finished!')

            # init centroids
            centroids = Y[center_label]

            iter = 10
            all_evals = []
            all_evals.append(['RI', 'Purity', 'NMI'])  #
            for step in range(1, iter + 1):
                part = np.zeros((M,K))
                for m in range(M):
                    for k in range(K):
                        # d = 0
                        # for r in range(Ps):
                        #     d += SGB(Y[m,r,:],centroids[k,r,:])
                        d = SWMDist(Y[m], centroids[k])
                        # d = Spatial_Weighted_Matrix_Dist(Y[m],centroids[k])
                        # print(d)
                        # temp = -2/(b-1)
                        # if d == 0:
                        #     part[m, k] = 1
                        #     for i in range(k):
                        #         part[m, i] = 0
                        #     break
                        # else:
                        #     d = np.power(d,temp)
                        #     part[m,k] = d
                        part[m, k] = d
                for m in range(M):
                    sumkr = np.sum(part[m, :])
                    part[m, :] = part[m, :] / sumkr

                # partition results
                P = np.zeros(M)
                in_cluster = [[] for _ in range(K)]
                for m in range(M):
                    indices = np.argmin(part[m,:])
                    P[m] = indices
                    in_cluster[indices].append(m)

                # print(part)
                # print(P)

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

                centroids = np.zeros((K,Ps,L))
                for k in range(K):
                    for n in in_cluster[k]:
                        centroids[k] += Y[n]
                    centroids[k] = centroids[k] / len(in_cluster[k])

            time_end = time.time()
            t = time_end - time_begin
            print("time", t)

            # save evaluation
            eval_file = file + "_SWMDist_init_center"
            book_name_xlsx = './result/' + file + '_SWMDist_evaluation.xlsx'
            sheet_name_xlsx = eval_file
            write_excel_xlsx(book_name_xlsx, sheet_name_xlsx, all_evals)


if __name__ == '__main__':
    main()
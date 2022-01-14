import numpy as np
import os
import h5py
import random
from evaluation import RandIndex
from evaluation import Purity
from evaluation import NMI
from sklearn.metrics.cluster import normalized_mutual_info_score
from save_to_excel import write_excel_xlsx
import time

# covariance matrix
def convariance_matrix(X):
    N = X.shape[0]
    R = X.shape[2]
    covs = np.zeros((N,R,R))
    for n in range(N):
        covs[n] = np.cov(X[n].T)
    return covs

def S_k(covs,K,p,in_cluster):
    R = covs.shape[1]
    S = np.zeros((K,R,p))
    for k in range(K):
        temp = in_cluster[k]
        cov_k = np.zeros((R,R))
        for n in temp:
            cov_k += covs[n]
        cov_k = cov_k / len(temp)
        eigenValues,eigenVectors = np.linalg.eig(cov_k)
        idx = eigenValues.argsort()[::-1] # 从大到小排序
        idx = idx[:p]
        # eigenValues = eigenValues[idx]
        S[k] = eigenVectors[:, idx]
    return S


def main():
    path = './data/'
    data_name = ['ArabicDigits_whole']
    # data_name = ['lp1','lp5','wafer298','net534','cricket', 'JapaneseVowels','ArticularyWord', 'char', 'uWaveGestureLibrary','ArabicDigits_whole']
    for file in data_name:
        filename = file + '.h5'
        filename = os.path.join(path, filename)
        f = h5py.File(filename, 'r')
        X = f['train_x'][:]
        # print(X.shape) # NxLxR
        y = f['train_y'][:]
        N = X.shape[0]
        K = len(np.unique(y))
        print("数据集",file)
        # p_s = [1,2]
        p_s = [int(X.shape[2]*0.5)]
        covs = convariance_matrix(X)

        all_evals = []
        for p in p_s:
            RI_value = 0
            purity_value = 0
            NMI_value = 0
            start = time.time()
            for step in range(10):
                part = [random.randint(0,K-1) for _ in range(N)]
                in_cluster = [[] for _ in range(K)]
                for n in range(N):
                    in_cluster[part[n]].append(n)
                S = S_k(covs, K, p, in_cluster)

                t = 0
                max_iter = 100
                prev_E = float("inf")
                while t <= max_iter:
                    # print(t)
                    t = t+1
                    E = 0
                    in_cluster = [[] for _ in range(K)]
                    for n in range(N):
                        errors = []
                        for k in range(K):
                            Sk = S[k]
                            Xi = X[n]
                            Yi = np.dot(Xi,Sk.dot(Sk.T))
                            errors.append(np.linalg.norm(Xi-Yi))
                        v = np.min(errors)
                        idx = np.argmin(errors)
                        E += v
                        part[n] = idx
                        in_cluster[idx].append(n)
                    for k in range(K):
                        if in_cluster[k] == []:
                            x = random.randint(0,N-1)
                            in_cluster[k].append(x)
                            part[x] = k
                    if E == prev_E:
                        break
                    prev_E = E

                    S = S_k(covs, K, p, in_cluster)
                # RI_value += RandIndex(part, y)
                # purity_value += Purity(part, y)
                # # NMI_value += NMI(part, y)
                # NMI_value += normalized_mutual_info_score(part,y)
            end = time.time()
            print("总花费时间为：",end-start)
            # print(RI_value / 10, purity_value / 10, NMI_value / 10)
            # temp = []
            # temp.append(RI_value/10)
            # temp.append(purity_value/10)
            # temp.append(NMI_value/10)
            # all_evals.append(temp)

        # save evaluation
        # eval_file = file + "_McPCA"
        # book_name_xlsx = './result/' + file + '_evaluation.xlsx'
        # sheet_name_xlsx = eval_file
        # write_excel_xlsx(book_name_xlsx, sheet_name_xlsx, all_evals)



if __name__ == '__main__':
    main()
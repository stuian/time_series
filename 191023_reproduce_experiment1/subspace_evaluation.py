import numpy as np

def subspace_evaluation(part,W):
    N = len(part)
    K = W.shape[0]
    R = W.shape[1]

    AJ = np.zeros((N,N)) # 聚类结果的邻接矩阵
    for i in range(N):
        for j in range(i+1,N):
            if part[i] == part[j]:
                AJ[i,j] = 1
                AJ[j,i] = 1
    Gamma = np.zeros(K)
    to_p = 0.25
    for k in range(K):
        Lap_matrix = Lap(N,R,W[k,:])
        Gamma[k] = norm
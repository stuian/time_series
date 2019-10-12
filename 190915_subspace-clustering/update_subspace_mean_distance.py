# C:C1,C2,...;[[],[],[],...]
# X:n x l x r
# center_label:peak

import numpy as np

def single_distance_between(a, b, r):
    # a,b是多元时间序列样本，r是变量
    a = a[:, r]
    b = b[:, r]
    product = np.dot(a, b)
    product = product / (np.sqrt(np.dot(a, a)) * np.sqrt(np.dot(b, b)))
    return 1 - product  # 越小越好


def getHA(part,in_cluster,N,K,R,center_label,X):
    # 求out_cluster
    out_cluster = []
    for i in range(K):
        temp = []
        for j in range(K):
            if j != i:
                temp = temp + in_cluster[j]
        out_cluster.append(temp)

    # 更新变量子空间，即HA
    HA = np.zeros((K, R))

    for k in range(K):
        length_ck = len(in_cluster[k])
        length_nk = len(out_cluster[k])

        for r in range(R):
            # 类间距离：平均距离法
            HD = 0
            for i in range(length_ck):
                for j in range(length_nk):
                    temp = single_distance_between(X[in_cluster[k][i]], X[out_cluster[k][j]], r)
                    HD += temp**2
            HD = HD / (length_ck*length_nk)

            pkr = 0
            for i in in_cluster[k]:
                pkr = pkr + single_distance_between(X[i],X[center_label[k]],r)
            pkr = np.sqrt(pkr) / length_ck # runningtime说明length_ck里出现了0

            # HA_up[k,r] = HD * (length_ck/N)
            # HA_down[k,r] = pkr * (length_ck/N)

            if pkr != 0:
                HA[k,r] = HD / pkr
            else:
                HA[k,r] = 0
        HA[np.isnan(HA)] = 0

    normalization = np.sum(HA,axis=1)
    for k in range(K):
        for r in range(R):
            HA[k,r] = HA[k,r] / normalization[k]
    return HA

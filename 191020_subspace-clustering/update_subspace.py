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


def update_subspace(in_cluster,out_cluster,center_label,X):
    K = len(in_cluster)
    R = X.shape[2]

    # 更新变量子空间，即HA
    W = np.zeros((K, R))

    for k in range(K):
        length_ck = len(in_cluster[k])
        length_nk = len(out_cluster[k])
        if length_ck > 1:
            for r in range(R):
                interdistance = np.zeros(length_ck * (length_ck - 1) // 2)
                outsidedistance = np.zeros(length_nk * (length_nk - 1) // 2)

                counter_i = 0
                for i in range(length_ck):
                    for j in range(i + 1, length_ck):
                        interdistance[counter_i] = single_distance_between(X[in_cluster[k][i]], X[in_cluster[k][j]],r)
                        counter_i += 1

                counter_i = 0
                for i in range(length_nk):
                    for j in range(i + 1, length_nk):
                        outsidedistance[counter_i] = single_distance_between(X[out_cluster[k][i]], X[out_cluster[k][j]],r)
                        counter_i += 1

                mean_distence_between = np.mean(interdistance)
                std_distence_between = np.std(interdistance)
                ukr2 = np.mean(outsidedistance)
                skr2 = np.std(outsidedistance)
                HD = np.sqrt(1 - np.sqrt((2 * std_distence_between * skr2) / (std_distence_between ** 2 + skr2 ** 2)) \
                    * np.exp(-(mean_distence_between - ukr2) ** 2 / (4 * (std_distence_between ** 2 + skr2 ** 2))))

                pkr = 0
                for i in in_cluster[k]:
                    pkr = pkr + single_distance_between(X[i],X[center_label[k]],r)**2
                pkr = np.sqrt(pkr) / length_ck # runningtime说明length_ck里出现了0

                W[k,r] = HD / pkr

            W[np.isnan(W)] = 1 / R
            sumkr = np.sum(W[k,:])
            if sumkr !=0 :
                W[k, :] = W[k,:] / sumkr
            else:
                W[k, :] = 1 / R
        else:
            W[k, :] = 1 / R
    return W

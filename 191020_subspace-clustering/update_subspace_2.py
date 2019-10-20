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


def getHA(part, in_cluster, N, K, R, center_label, X):
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
    mean_distence_between = np.zeros((K, R))
    std_distence_between = np.zeros((K, R))

    for k in range(K):
        length_ck = len(in_cluster[k])
        length_nk = len(out_cluster[k])
        for r in range(R):
            interdistance = np.zeros(length_ck * (length_ck - 1) // 2)
            outsidedistance = np.zeros(length_ck * length_nk)

            counter_i = 0
            for i in range(length_ck):
                for j in range(i + 1, length_ck):
                    interdistance[counter_i] = single_distance_between(X[in_cluster[k][i]], X[in_cluster[k][j]], r)
                    counter_i += 1

            counter_i = 0
            for i in range(length_ck):
                for j in range(length_nk):
                    outsidedistance[counter_i] = single_distance_between(X[in_cluster[k][i]], X[out_cluster[k][j]],r)
                    counter_i += 1

            mean_distence_between[k, r] = np.mean(interdistance)
            std_distence_between[k, r] = np.std(interdistance)
            ukr2 = np.mean(outsidedistance)
            skr2 = np.std(outsidedistance)
            HD = np.sqrt(1 - np.sqrt((2 * std_distence_between[k, r] * skr2) / (std_distence_between[k, r] ** 2 + skr2 ** 2)) \
                * np.exp(-(mean_distence_between[k, r] - ukr2) ** 2 / (4 * (std_distence_between[k, r] ** 2 + skr2 ** 2))))

            pkr = 0
            for i in in_cluster[k]:
                pkr = pkr + single_distance_between(X[i], X[center_label[k]], r)
            pkr = np.sqrt(pkr) / length_ck  # runningtime说明length_ck里出现了0

            if pkr != 0:
                HA[k, r] = HD / pkr
            else:
                HA[k, r] = 0
        HA[np.isnan(HA)] = 0

    normalization = np.sum(HA, axis=1)
    for k in range(K):
        for r in range(R):
            HA[k, r] = HA[k, r] / normalization[k]
    return HA
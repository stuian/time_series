# C:C1,C2,...;[[],[],[],...]
# X:n x l x r
# center_label:peak

import numpy as np

def single_distance_between(a,b,r):
    # a,b是多元时间序列样本，r是变量
    a = a[:,r]
    b = b[:,r]
    product = np.dot(a, b)
    product = product / (np.sqrt(np.dot(a, a)) * np.sqrt(np.dot(b, b)))
    return 1 - product  # 越小越好

def getHA(part,in_cluster,distance,centerLabel,W,X):
    """
    update_HA
    :param part:cluster assignment
    :param num_incluster:簇中的序列数
    :param distance:N,K,R
    :param centerLabel:
    :param W:
    :return:
    """
    N,K,R = np.shape(distance)

    out_cluster = [[] for i in range(K)]

    for k in range(K):
        if len(in_cluster[k]) > 0:
            for i in range(N):
                if part[i] != k:
                    out_cluster[k].append(i)

    HA = np.zeros((K,R))
    mean_distence_between = np.zeros((K,R))
    std_distence_between = np.zeros((K,R))

    for k in range(K):
        if len(in_cluster[k]) > 1:
            length_ck = len(in_cluster[k])
            length_nk = len(out_cluster[k])
            for r in range(R):
                interdistance = np.zeros(length_ck*(length_ck-1)/2)
                interdistance_a = np.zeros((length_ck,length_ck))
                outsidedistance = np.zeros(length_nk*(length_nk-1)/2)
                outsidedistance_a = np.zeros((length_nk,length_nk))

                counter_i = 1
                for i in range(length_ck):
                    for j in range(i+1,length_ck):
                        interdistance[counter_i] = single_distance_between(X[in_cluster[k][i]],X[in_cluster[k][j]],r)
                        counter_i += 1

                counter_i = 1
                for i in range(length_nk):
                    for j in range(i+1,length_nk):
                        outsidedistance[counter_i] = single_distance_between(X[out_cluster[k][i]],X[out_cluster[k][j]],r)
                        counter_i += 1
                mean_distence_between[k,r] = np.mean(interdistance)
                std_distence_between[k,r] = np.std(interdistance)
                ukr2 = np.mean(outsidedistance)
                skr2 = np.std(outsidedistance)
                HD = np.sqrt(1 - np.sqrt((2 * std_distence_between[k, r] * skr2) / (std_distence_between[k, r]**2 + skr2**2))\
                             * np.exp(-(mean_distence_between[k, r] - ukr2)**2 / (4 * (std_distence_between[k, r]**2 + skr2**2))))
                pkr = 0






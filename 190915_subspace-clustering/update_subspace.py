# C:C1,C2,...;[[],[],[],...]
# X:n x l x r
# center_label:peak

import numpy as np

def getHA(part,in_cluster,distance,centerLabel,W):
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

    for k in range(K):
        if len(in_cluster[k]) > 1:
            length_ck = len(in_cluster[k])
            length_nk = len(out_cluster[k])
            for r in range(R):
                interdistance = np.zeros(())




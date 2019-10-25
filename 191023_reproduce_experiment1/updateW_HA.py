import numpy as np

def update_subspace(in_cluster,out_cluster,center_label,single_distance_between):
    K = len(in_cluster)
    R = single_distance_between.shape[2]

    # 更新变量子空间
    # 方法一：HA；比较簇内样本距离分布和簇外样本距离分布
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
                        interdistance[counter_i] = single_distance_between[in_cluster[k][i], in_cluster[k][j],r]
                        counter_i += 1

                counter_i = 0
                for i in range(length_nk):
                    for j in range(i + 1, length_nk):
                        outsidedistance[counter_i] = single_distance_between[out_cluster[k][i], out_cluster[k][j],r]
                        counter_i += 1

                mean_distence_between = np.mean(interdistance)
                std_distence_between = np.std(interdistance)
                ukr2 = np.mean(outsidedistance)
                skr2 = np.std(outsidedistance)
                HD = np.sqrt(1 - np.sqrt((2 * std_distence_between * skr2) / (std_distence_between ** 2 + skr2 ** 2)) \
                    * np.exp(-(mean_distence_between - ukr2) ** 2 / (4 * (std_distence_between ** 2 + skr2 ** 2))))

                pkr = 0
                for i in in_cluster[k]:
                    pkr = pkr + single_distance_between[i,center_label[k],r]**2
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

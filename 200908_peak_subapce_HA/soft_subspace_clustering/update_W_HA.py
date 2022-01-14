import numpy as np

def update_subspace(in_cluster,out_cluster,center_label,W,single_distance_between):
    K = len(in_cluster)
    R = single_distance_between.shape[2]

    # 更新变量子空间
    # 方法：比较簇内样本距离分布和簇外样本距离分布
    HA = np.zeros((K, R))
    MMD = np.zeros((K, R))
    pkr = np.zeros((K, R))

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

                # combine HA with MMD
                # interdistance = np.zeros((length_ck, length_ck))
                # outsidedistance = np.zeros((length_nk, length_nk))
                # betweendistance = np.zeros((length_ck, length_nk))
                #
                # for i in range(length_ck):
                #     for j in range(i + 1, length_ck):
                #         interdistance[i, j] = single_distance_between[in_cluster[k][i], in_cluster[k][j],r]**2
                #         interdistance[j, i] = interdistance[i, j]
                # interD = np.sum(interdistance) / (length_ck * (length_ck - 1))
                #
                # for i in range(length_nk):
                #     for j in range(i + 1, length_nk):
                #         outsidedistance[i, j] = single_distance_between[out_cluster[k][i], out_cluster[k][j],r]**2
                #         outsidedistance[j, i] = outsidedistance[i, j]
                # outerD = np.sum(outsidedistance) / (length_nk * (length_nk - 1))
                #
                # for i in range(length_ck):
                #     for j in range(length_nk):
                #         betweendistance[i, j] = single_distance_between[in_cluster[k][i], out_cluster[k][j],r]**2
                # betweenD = np.sum(betweendistance) / (length_ck * length_nk)
                #
                # MMD[k, r] = abs(interD + outerD - 2 * betweenD)

                # for i in in_cluster[k]:
                #     pkr[k, r] = pkr[k, r] + single_distance_between[i, center_label[k], r] ** 2
                # pkr[k, r] = np.sqrt(pkr[k, r]) / length_ck  # runningtime说明length_ck里出现了0

            # alpha = 3 # 2
            # beta = 0.2 # 0.05
            # MMD[k, :] = 1.0 / (np.power(MMD[k, :], 1 / alpha))
            # MMD[k, :] = MMD[k, :] / np.sum(MMD[k, :])
            # pkr[k, :] = pkr[k, :] / np.sum(pkr[k, :])
            # HA[k, :] = pkr[k, :] * MMD[k, :]
            # HA[k, :] = np.exp(-HA[k, :] / beta)
            # lamda = 0.5
            # HA[k, :] = lamda * W[k, :] + (1 - lamda) * HA[k, :]

                pkr = 0
                for i in in_cluster[k]:
                    pkr = pkr + np.power(single_distance_between[i,center_label[k],r],2)
                pkr = np.sqrt(pkr) / length_ck # runningtime说明length_ck里出现了0
                #
                HA[k,r] = HD / (pkr+1e-5)
                #
                # lamda = 0.5
                # HA[k, r] = lamda * W[k, r] + (1 - lamda) * HA[k, r]
            HA[np.isnan(HA)] = 1 / R
            sumkr = np.sum(HA[k,:])
            if sumkr !=0 :
                HA[k, :] = HA[k,:] / sumkr
            else:
                HA[k, :] = 1 / R
        else:
            HA[k, :] = 1 / R
    return HA
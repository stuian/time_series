# 更新变量子空间后，峰值也会发生变化

def KNN_SMin(DMax, SMin, k, distance):
    if DMax == SMin:
        return DMax
    else:
        peak = -1
        temp_distance = distance[DMax]
        sort_points = sorted(enumerate(temp_distance), key=lambda x: x[1])
        sort_points = sort_points[:k]
        D_to_SMin = {}
        dist = float("inf")
        for i in range(k):
            if dist < distance[sort_points[i][0]][SMin]:
                peak = sort_points[i][0]
        return peak

def update_peak(center_label,in_cluster,density,distance):
    K = len(in_cluster)
    for k in range(K):
        # 对簇Ck中每条序列样本计算其局部密度𝜌𝑖 ，以及到簇内其它序列的距离之和SD𝑖
        Ck_density = {}
        Ck_sd = {}
        for i in in_cluster[k]:
            Ck_density[i] = density[i]
            Ck_sd[i] = 0
            for j in in_cluster[k]:
                if j != i:
                    Ck_sd[i] = Ck_sd[i] + distance[i][j]
        sorted_density = sorted(Ck_density.items(), key=lambda x: x[1], reverse=True)
        DMax = sorted_density[0][0]
        sorted_sd = sorted(Ck_sd.items(), key=lambda x: x[1])
        SMin = sorted_sd[0][0]
        m = 20
        center_label[k] = KNN_SMin(DMax, SMin, m, distance)
    return center_label


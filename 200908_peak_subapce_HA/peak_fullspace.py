import h5py
import numpy as np
import os
from densityPeakRNN import densityPeakRNN
from evaluation import RandIndex
from evaluation import Purity
from evaluation import NMI
from save_to_excel import write_excel_xlsx


def subspace_evaluation3(in_cluster,center_label,single_distance_between): # 目标函数值越来越大
    K = len(center_label)
    R = single_distance_between.shape[2]

    pkr = 0
    for k in range(K):
        c = int(center_label[k])
        for n in in_cluster[k]:
            temp = 0
            for r in range(R):
                temp += single_distance_between[c, n, r]
            pkr += temp
    return pkr

# 1、get data
choice = 1
path = './data/'
if choice == 1:
    data_name = ["uWaveGestureLibrary"]# ['lp1','lp2','lp3','lp4','lp5']
elif choice == 2:
    data_name = ['pen','uWaveGestureLibrary']
elif choice == 3:
    data_name = ['wafer298','net534','vowels','ArticulographData','char300']
for file in data_name:
    if choice == 1:
        filename = file + '.h5'
        filename = os.path.join(path,filename)
        f = h5py.File(filename,'r')
        X = f['train_x'][:]
        print(X.shape)
        y = f['train_y'][:] # 1,2,3... np.array
    elif choice == 3:
        label_name = file + "_label.npy"
        y = np.load(os.path.join("./result", label_name))

    print("%s数据集进行全空间聚类..." % file)

    # 获取正常（没有变量权值）的距离矩阵
    PATH = './result/' + file + '_dist.npy'
    single_distance_between = np.load(PATH)

    eval_file = file + '_fullspace'

    # 2、初始化峰值
    K = len(np.unique(y))
    print('聚类簇数：', K)
    N = single_distance_between.shape[0]
    R = single_distance_between.shape[2]

    # k:求某点时的密度，KNN的k值
    all_evals = []
    for k in range(1,11):
        center_label = densityPeakRNN(k, K, single_distance_between)

        # 全空间
        # W = np.ones((K,R))
        # W = W / R

        part = np.zeros(N)
        in_cluster = [[] for _ in range(K)]
        for n in range(N):
            minDist = float('inf')
            index = -1
            for i in range(len(center_label)):
                if n == int(center_label[i]):
                    index = i
                    break
                else:
                    temp = 0
                    for r in range(R):
                        temp += single_distance_between[n, int(center_label[i]), r]*(1/R)
                    if temp < minDist:
                        minDist = temp
                        index = i

            ck = index
            part[n] = ck
            in_cluster[ck].append(n)

        # cost = sihouette_coefficient(part, in_cluster, center_label, W, single_distance_between)
        # cost = subspace_evaluation3(in_cluster,center_label,single_distance_between)

        temp = []
        RI_value = RandIndex(part, y)
        purity_value = Purity(part, y)
        NMI_value = NMI(part, y)
        temp.append(RI_value)
        temp.append(purity_value)
        temp.append(NMI_value)
        # temp.append(cost)
        all_evals.append(temp)
        # print(RI_value,purity_value,NMI_value)

    # save results
    book_name_xlsx = './result/' + file + '_evaluation.xlsx'
    sheet_name_xlsx = eval_file
    write_excel_xlsx(book_name_xlsx, sheet_name_xlsx, all_evals)
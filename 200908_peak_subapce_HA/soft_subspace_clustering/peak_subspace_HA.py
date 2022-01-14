import h5py
import numpy as np
import os
import scipy.io as scio
from densityPeakRNN import densityPeakRNN
from cluster import series_to_centers
from update_W_HA import update_subspace
# from updateW_MHA import update_subspace
from update_RNNpeak import update_peak
from evaluation import RandIndex
from evaluation import Purity
from evaluation import NMI
from save_to_excel import write_excel_xlsx
import time

def subspace_evaluation(in_cluster,center_label,single_distance_between,W):
    K = len(center_label)
    R = single_distance_between.shape[2]

    pkr = 0
    for k in range(K):
        c = int(center_label[k])
        for n in in_cluster[k]:
            temp = 0
            for r in range(R):
                temp += np.power(single_distance_between[c, n, r],2)*W[k,r]
            temp = np.sqrt(temp)
            pkr += temp
    return pkr


def peak_subspace(y,file,single_distance_between):
    # 2、初始化峰值和随机子空间
    K = len(np.unique(y))
    print('聚类簇数：', K)
    N = single_distance_between.shape[0]
    R = single_distance_between.shape[2]

    # 2.1 初始化峰值
    # k:求某点的密度中KNN的k值
    eval_file = file + "_HA"
    # count = 0
    # center_label = []
    # while count != K:
    #     temp = np.random.randint(0,N)  # 前闭后开
    #     if temp not in center_label:
    #         count += 1
    #         center_label.append(temp)

    start = time.time()
    k = 5
    center_label = densityPeakRNN(k, K, single_distance_between) # return center_label
    end = time.time()
    print("initial center_label: ", center_label,", cost time :",end-start)
    # count = 0
    # center_label = []
    # while count != K:
    #     temp = np.random.randint(0, N)  # 前闭后开
    #     if temp not in center_label:
    #         count += 1
    #         center_label.append(temp)

    # 2.2 随机子空间
    # np.random.seed(0)
    # W = np.random.random((K,R))
    # s = W.sum(axis=1)
    # # 标准化，每一行相加等于1
    # for i in range(K):
    #     W[i] = W[i] / s[i]
    W = np.ones((K, R))
    W = W / R

    # 迭代循环
    iter = 15
    all_evals = []
    all_evals.append(['RI','Purity','NMI','Cost']) #
    start = time.time()
    for step in range(1, iter + 1):
        # 3、样本分配到簇
        # 该步骤要用到变量子空间
        part = np.zeros(N)
        in_cluster = [[] for _ in range(K)]
        for n in range(N):
            ck = series_to_centers(single_distance_between, n, center_label, W)
            part[n] = ck
            in_cluster[ck].append(n)

        out_cluster = []
        for i in range(K):
            temp = []
            for j in range(K):
                if j != i:
                    temp = temp + in_cluster[j]
            out_cluster.append(temp)

        # # 评价子空间结果
        # cost = subspace_evaluation(in_cluster,center_label,single_distance_between,W)
        # # print("the cost of iter %d: " % step,cost)
        #
        # # evaluation
        # temp = []
        # RI_value = RandIndex(part, y)
        # purity_value = Purity(part, y)
        # NMI_value = NMI(part, y)
        # print(RI_value,purity_value,NMI_value,cost)
        # temp.append(RI_value)
        # temp.append(purity_value)
        # temp.append(NMI_value)
        # temp.append(cost)
        # all_evals.append(temp)

        # 4、更新子空间和峰值
        # 4.1 更新子空间 # HA
        # plt.figure(12)
        # plt.subplot(121)
        # plt.bar(range(R),W[0,:])
        # plt.subplot(122)
        # plt.bar(range(R), W[1, :])
        # plt.show()
        W = update_subspace(in_cluster,out_cluster,center_label,W,single_distance_between)
        # W = update_subspace(in_cluster,out_cluster,center_label,X,single_distance_between)
        # print(W)

        # 4.2 更新峰值
        # 子空间峰值更新
        center_label = update_peak(k,in_cluster, center_label, W, single_distance_between)
        # print(center_label)

        end = time.time()
        if step == 1:
            print("迭代一次花费%ds" % (end-start))
    print("总花费时间%ds" % (end-start))

    # save results
    # PATH = './result/' + file + '_part.npy'
    # np.save(PATH, part)

    # save evaluation
    # book_name_xlsx = './result/' + file + '_evaluation.xlsx'
    # sheet_name_xlsx = eval_file
    # write_excel_xlsx(book_name_xlsx, sheet_name_xlsx, all_evals)

    # return all_evals

if __name__ == '__main__':
    # 1、get data
    choice = 1
    path = './data/'
    if choice == 1:
        data_name = ['net534']
        # 'Libras Movement','lp5', 'ArticularyWord','cricket_data', 'uWaveGesture', 'BCI','EEG'
    elif choice == 2:
        data_name = ['pen', 'uWaveGestureLibrary']
    elif choice == 3:
        data_name = ['AUSLAN'] #'net534'，'wafer298', 'vowels', 'ArticulographData','char300'
    for file in data_name:
        if choice == 1:
            filename = file + '.h5'
            filename = os.path.join(path, filename)
            f = h5py.File(filename, 'r')
            X = f['train_x'][:]
            # print(X.shape)
            y = f['train_y'][:]  # 1,2,3... np.array
        elif choice == 3:
            # filename = file + '.mat'
            # filename = os.path.join(path, filename)
            # myfile = scio.loadmat(filename)
            # for element in myfile['data']:
            #     X = element
            # print(X[3][:].shape)
            label_name = file + "_labels.npy"
            y = np.load(os.path.join("./result", label_name))

        print("%s数据集进行子空间聚类..." % file)
        # print('数据集大小', X.shape)

        # filename = file + '_subspace'

        # 获取正常（没有变量权值）的距离矩阵
        PATH = './result/' + file + '_dist.npy'
        single_distance_between = np.load(PATH)



        peak_subspace(y,file,single_distance_between)


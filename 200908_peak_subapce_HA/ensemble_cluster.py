import h5py
import os
import numpy as np
from ensemble_framework import center_value
from ensemble_framework import neigbor_matrix
from evaluation import RandIndex
from evaluation import Purity
from evaluation import NMI
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from save_to_excel import write_excel_xlsx
import warnings
warnings.filterwarnings("ignore")



def main():
    choice = 1
    path = './data/'
    if choice == 1:
        data_name = ['uWaveGestureLibrary']#['cricket','net534', 'uWaveGesture', 'ArticularyWord']
                    # 'lp5','cricket_data','Libras Movement', 'uWaveGesture', 'ArticularyWord','EEG','BCI'
    elif choice == 2:
        data_name = ['pen', 'uWaveGestureLibrary']
    elif choice == 3:
        data_name = ["char"] #"CMUsubject16",'ECG','ArabicDigits','AUSLAN','wafer298', 'net534', 'vowels', 'ArticulographData', 'char300'
    for file in data_name:
        if choice == 1:
            filename = file + '.h5'
            filename = os.path.join(path, filename)
            f = h5py.File(filename, 'r')
            X = f['train_x'][:]
            print(X.shape)
            y = f['train_y'][:]  # 1,2,3... np.array
        elif choice == 3:
            label_name = file + "_labels.npy"
            y = np.load(os.path.join("./result", label_name))

        print("%s数据集进行子空间集成聚类..." % file)
        # print('数据集大小', X.shape)

        # filename = file + '_subspace'
        eval_file = file + "_ensemble_rw"

        # 获取正常（没有变量权值）的距离矩阵
        PATH = './result/' + file + '_dist.npy'
        single_distance_between = np.load(PATH)

        N = single_distance_between.shape[0]
        R = single_distance_between.shape[2]
        K = len(np.unique(y))

        all_evals = []
        evals = []
        results = []
        for iter in range(1):
            print(iter)
            # B = 30

            parts = np.load(os.path.join('./result/', file + '_parts_%d.npy' % iter))

            # 三个一组

            count = 0
            for i in range(10):
                concensus_matrix = np.zeros((N, N))
                for j in range(3):
                    neigbor = neigbor_matrix(parts[count])
                    concensus_matrix += neigbor
                    count += 1
                concensus_matrix = concensus_matrix / 3

                center = center_value(concensus_matrix)
                af = AffinityPropagation(affinity='precomputed', damping=0.5, preference=center).fit(
                    concensus_matrix)
                labels = af.labels_
                temp = []
                RI_value = RandIndex(labels, y)
                purity_value = Purity(labels, y)
                NMI_value = NMI(labels, y)
                temp.append(RI_value)
                temp.append(purity_value)
                temp.append(NMI_value)
                results.append(temp)
                print(temp)
                #
                # pred_y = SpectralClustering(n_clusters=K, assign_labels="discretize", random_state=3,
                #                             affinity='precomputed').fit_predict(concensus_matrix)
                #
                # temp = []
                # RI_value = RandIndex(y, pred_y)
                # purity_value = Purity(y, pred_y)
                # NMI_value = NMI(y, pred_y)
                # temp.append(RI_value)
                # temp.append(purity_value)
                # temp.append(NMI_value)
                # all_evals.append(temp)
                # print(temp)
                #
                # agg = AgglomerativeClustering(n_clusters=K, affinity='precomputed', linkage='complete')
                # predicted_labels = agg.fit_predict(concensus_matrix)
                #
                # temp = []
                # RI_value = RandIndex(y, predicted_labels)
                # purity_value = Purity(y, predicted_labels)
                # NMI_value = NMI(y, predicted_labels)
                # temp.append(RI_value)
                # temp.append(purity_value)
                # temp.append(NMI_value)
                # evals.append(temp)
                # print(temp)

            # save evaluation
            eval_file = file + "_AP"
            book_name_xlsx = './result/' + file + '_evaluation.xlsx'
            sheet_name_xlsx = eval_file
            write_excel_xlsx(book_name_xlsx, sheet_name_xlsx, results)

            # # save evaluation
            # eval_file = file + "_SP"
            # book_name_xlsx = './result/' + file + '_evaluation.xlsx'
            # sheet_name_xlsx = eval_file
            # write_excel_xlsx(book_name_xlsx, sheet_name_xlsx, all_evals)
            #
            # # save evaluation
            # eval_file = file + "_HC"
            # book_name_xlsx = './result/' + file + '_evaluation.xlsx'
            # sheet_name_xlsx = eval_file
            # write_excel_xlsx(book_name_xlsx, sheet_name_xlsx, evals)

if __name__ == '__main__':
    main()
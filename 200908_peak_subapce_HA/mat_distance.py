import scipy.io as scio
import numpy as np
import os
import time

def SBD(x,y):
    # 越小越相似
    y_ = [i for i in reversed(y)]
    CC = np.convolve(x,y_)
    s = np.linalg.norm(x) * np.linalg.norm(y)
    if s == 0 :
        value = 0
    else:
        NCC = CC / s
        index = np.argmax(NCC)
        value = NCC[index]
    dist = 1 - value
    return dist

def main():
    path = './data'
    file = 'ECG'
    PATH = './result/'
    distancename = file + '_dist.npy'
    distance_path = os.path.join(PATH, distancename)
    labelsname = file + "_labels.npy"
    labels_path = os.path.join(PATH,labelsname)
    begin_time = time.time()
    filename = file + '.mat'
    data = scio.loadmat(os.path.join(path,filename))
    print(data['mts'][0, 0]['train'].shape)
    print(data['mts'][0, 0]['trainlabels'].shape)
    labels = []
    N = data['mts'][0, 0]['trainlabels'].shape[0]
    for i in range(N):
        labels.append(data['mts'][0, 0]['trainlabels'][i][0])
    np.save(labels_path,labels)
    # print(data['mts'][0, 0]['testlabels'][0][0])
    for element in data['mts'][0, 0]['train']:
        R = element[1].shape[0]
        print(R)
        single_distance_between = np.zeros((N, N, R))
        for i in range(N):
            for j in range(i):
                for r in range(R):
                    single_distance_between[i, j, r] = SBD(element[i][r,:], element[j][r,:])
                    single_distance_between[j, i, r] = single_distance_between[i, j, r]
        print(single_distance_between.shape)
        print('完成计算！')
        np.save(distance_path, single_distance_between)
        print('已保存距离结果')
        end_time = time.time()
        print("time:", end_time - begin_time)



if __name__ == '__main__':
    main()
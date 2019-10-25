import numpy as np
import h5py

def series_to_centers(single_distance_between, x, center_label, W):
    minDist = float('inf')
    index = -1
    R = W.shape[1]
    for i in range(len(center_label)):
        if x == center_label[i]:
            return i
        else:
            temp = 0
            for r in range(R):
                temp += np.power(single_distance_between[x,center_label[i],r],2) * W[i,r]
            temp = np.sqrt(temp)
            if temp < minDist:
                minDist = temp
                index = i
    return index

def main():
    file = h5py.File('E:\\Jade\\time_series\\190808_MTS-clustering\\cricket_data.h5', 'r')
    X = file['train_x'][:]
    y = file['train_y'][:]
    r = X.shape[2];k = len(np.unique(y))
    # 变量子空间初始化
    variable_weight = globals()
    for i in range(k):
        variable_weight['s%s' % str(k + 1)] = []  # s1,...,sk
        for j in range(r):
            variable_weight['s%s' % str(k + 1)].append(1 / r)
    # print(variable_weight['s%s' % str(k + 1)])



if __name__ == '__main__':
    main()
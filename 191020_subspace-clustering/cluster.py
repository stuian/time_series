import numpy as np
import h5py

# 各序列到各中心的距离
def inner_product(a,b):
    product = np.dot(a,b)
    product = product / (np.sqrt(np.dot(a,a)) * np.sqrt(np.dot(b,b)))
    return 1-product # 越小越好

def multi_similarity(x,y,weight):
    value = 0
    for i in range(x.shape[1]):
        value = value + weight[i] * (inner_product(x[:,i],y[:,i]))**2
    return np.sqrt(value)

def series_to_centers(X,x,center_label,weight):
    minDist = float('inf')
    index = -1
    for i in range(len(center_label)):
        if x == center_label[i]:
            return i
        else:
            temp = multi_similarity(X[center_label[i]],X[x],weight[i])
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
import h5py
import numpy as np
import os
from density_peak import densityPeakRNN

# 1、get Robot Execution Failures lp1-5 data
path = 'E:\\Jade\\time_series\\190808_MTS-clustering'
data_name = ['Robot Execution Failures lp1', 'Robot Execution Failures lp2', 'Robot Execution Failures lp3','Robot Execution Failures lp4','Robot Execution Failures lp5']
for file in data_name:
    filename = file + '.h5'
    filename = os.path.join(path,filename)
    f = h5py.File(filename,'r')
    X = f['train_x'][:]
    y = f['train_y'][:] # 1,2,3... np.array

    # 获取正常（没有变量权值）的距离矩阵
    PATH = './data/' + file + '_distance.npy'
    if os.path.exists(PATH):
        D = np.load(PATH)
    else:


    # 2、初始化峰值和随机子空间
    K = len(np.unique(y))
    N = X.shape[0]
    L = X.shape[1]
    R = X.shape[2]


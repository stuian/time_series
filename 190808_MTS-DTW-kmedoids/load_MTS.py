import numpy as np
import h5py

# 正规化数据集

xleft = np.genfromtxt('../190806_DTW_multi_time_series/dataSets/Cricket/xleft.txt',delimiter=' ')
label = xleft[:,0]
xleft = xleft[:,1:]
yleft = np.genfromtxt('../190806_DTW_multi_time_series/dataSets/Cricket/yleft.txt',delimiter=' ')[:,1:]
zleft = np.genfromtxt('../190806_DTW_multi_time_series/dataSets/Cricket/zleft.txt',delimiter=' ')[:,1:]
xright = np.genfromtxt('../190806_DTW_multi_time_series/dataSets/Cricket/xright.txt',delimiter=' ')[:,1:]
yright = np.genfromtxt('../190806_DTW_multi_time_series/dataSets/Cricket/yright.txt',delimiter=' ')[:,1:]
zright = np.genfromtxt('../190806_DTW_multi_time_series/dataSets/Cricket/zright.txt',delimiter=' ')[:,1:]
# print(xleft.shape)
# print(yleft.shape)
# print(zleft.shape)
# print(xright.shape)
# print(yright.shape)
# print(zright.shape) # (180, 1198-1)

m = xleft.shape[0]
n = xleft.shape[1]
num = 6
data = np.concatenate((xleft,yleft,zleft,xright,yright,zright),axis=1)
data = data.reshape(m,num,n) # 样本数量；变量数目；样本长度
data = data.transpose(0,2,1)
# print(data.shape)  # (180, 1197, 6)

file = h5py.File('cricket_data.h5','w')
file.create_dataset('train_x',data = data)
file.create_dataset('train_y',data = label)
file.close()

# file = h5py.File('cricket_data.h5','r')
# train_x = file['train_x'][:]
# train_y = file['train_y'][:]



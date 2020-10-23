import h5py
import numpy as np
import scipy.io as sio
import os
# from tslearn.preprocessing import TimeSeriesScalerMeanVariance

path = './data/'
file = 'wafer298'
filename = file + '.mat'
filename = os.path.join(path, filename)
data = sio.loadmat(filename)['data'].ravel()
# X = [element[:] for element in X]

labels = sio.loadmat(filename)['trueLabel'].ravel()

max_length = float('-inf')
m = data[0].shape[0]
for element in data:
    temp = element.shape[1]
    if temp > max_length:
        max_length = temp
# print(max_length)

count = 0
for element in data:
    # element = TimeSeriesScalerMeanVariance(mu=0.0, std=1.0).fit_transform(element)
    if count == 0:
        X = np.zeros((m,max_length)) # 变量x长度
        X[:m,:element.shape[1]] = element # [:,:,0]
        count += 1
    else:
        temp = np.zeros((m,max_length))
        temp[:m, :element.shape[1]] = element # [:,:,0]
        X = np.concatenate((X, temp), axis=1)
        count += 1
X = X.reshape(count, m, max_length)  # 样本数量；变量数目；样本长度
print(X.shape)
X = X.transpose(0, 2, 1)
print(X.shape)

save_file = file + '.h5'
save_file = os.path.join(path, save_file)
f = h5py.File(save_file,'w')
f.create_dataset('train_x',data = X)
f.create_dataset('train_y',data = labels)
f.close()

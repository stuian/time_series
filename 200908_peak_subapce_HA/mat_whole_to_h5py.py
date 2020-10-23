import h5py
import numpy as np
import scipy.io as scio
import os

def main():
    path = './data'
    file = 'CMUsubject16'
    filename = file + '.mat'
    data = scio.loadmat(os.path.join(path, filename))

    labels = []
    train_N = data['mts'][0, 0]['trainlabels'].shape[0]
    test_N = data['mts'][0, 0]['testlabels'].shape[0]
    N = train_N + test_N
    print(N)

    for i in range(N):
        if i < train_N:
            labels.append(data['mts'][0, 0]['trainlabels'][i][0])
        else:
            labels.append(data['mts'][0, 0]['testlabels'][i - train_N][0])

    for element in data['mts'][0, 0]['train']:
        R = element[1].shape[0]
        print(R)
        train = element
    for element in data['mts'][0, 0]['test']:
        test = element

    max_length = float('-inf')
    for i in range(N):
        if i < train_N:
            temp = train[i].shape[1]
        else:
            temp = test[i - train_N].shape[1]
        if temp > max_length:
            max_length = temp

    count = 0
    for i in range(N):
        element = train[i] if i < train_N else test[i - train_N]
        if count == 0:
            X = np.zeros((R, max_length))  # 变量x长度
            X[:R, :element.shape[1]] = element
            count += 1
        else:
            temp = np.zeros((R, max_length))
            temp[:R, :element.shape[1]] = element  # [:,:,0]
            X = np.concatenate((X, temp), axis=1)
            count += 1
    X = X.reshape(count, R, max_length)  # 样本数量；变量数目；样本长度
    print(X.shape)
    X = X.transpose(0, 2, 1)
    print(X.shape)

    save_file = file + '.h5'
    save_file = os.path.join(path, save_file)
    f = h5py.File(save_file, 'w')
    f.create_dataset('train_x', data=X)
    f.create_dataset('train_y', data=labels)
    f.close()

if __name__ == '__main__':
    main()
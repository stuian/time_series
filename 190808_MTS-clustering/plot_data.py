import matplotlib.pyplot as plt
import h5py
import numpy as np

def Z_normalization(x,mu,sigma):
    x = (x - mu) / sigma
    return x

def main():
    file = h5py.File('cricket_data.h5', 'r')
    X = file['train_x'][:]
    y = file['train_y'][:]
    labels = np.unique(y)
    plot_samples = []
    for i in labels:
        for ind,j in enumerate(y):
            if j == i:
                plot_samples.append(ind)
                break

    plt.figure(figsize=(20,10),dpi=80)
    # 1
    plt.figure(1)
    for num in range(6):
        plt.subplot(231+num)
        for i in range(X[plot_samples[num]].shape[1]):
            plt.title('label %s' % str(num+1))
            plt.plot(X[plot_samples[num]][:,i])
    # 2
    plt.figure(figsize=(20, 10), dpi=80)
    plt.figure(2)
    for num in range(6):
        plt.subplot(231 + num)
        for i in range(X[plot_samples[6+num]].shape[1]):
            plt.title('label %s' % str(num + 7))
            plt.plot(X[plot_samples[6+num]][:, i])
    plt.show()

if __name__ == '__main__':
    main()
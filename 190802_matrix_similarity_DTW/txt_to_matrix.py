import numpy as np
import os

abnormal_path = '../data/multi_var/wafer/abnormal'

def two_matrix_samples():
    X1 = np.genfromtxt(os.path.join(abnormal_path,'1.txt'),delimiter=' ')
    X2 = np.genfromtxt(os.path.join(abnormal_path,'2.txt'), delimiter=' ')

    return X1,X2

if __name__ == '__main__':
    X1,X2 = two_matrix_samples()
    print(X1[:,0])
    print(len(X1[:,0]))
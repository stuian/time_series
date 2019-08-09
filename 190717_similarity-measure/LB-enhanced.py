import numpy as np
from random import randint
from DTW import LB_Keogh

def LB_Enhanced(A,B,W,V,D):
    """
    a new method to lower bound DTW
    :param A: time series;A[0]-A[59]
    :param B: time series
    :param W: warping window;w=0.1xL=6
    :param V: speed-tightness parameter;20
    :param D: current distance to NN
    :return:
    """
    res = (A[0]-B[0])**2 + (A[-1]-B[-1])**2
    n_bands = min(L/2,V)
    # 1、do L,R bands loop
    for i in range(2,n_bands+1):
        minL = (A[i-1]-B[i-1])**2
        minR = (A[L-i]-B[L-i])**2
        for j in range(max(1,i-W),i-1):
            minL = min(minL,(A[i-1]-B[j-1])**2)
            minL = min(minL,(A[j-1]-B[i-1])**2)
            minR = min(minR,(A[L-i]-B[L-j])**2)
            minR = min(minR,(A[L-j]-B[L-i])**2)
        res = res + minL + minR
    if res >= D:
        return float('inf')
    #2、LB_keogh
    temp = LB_Keogh(A[n_bands:L-n_bands],B[n_bands:L-n_bands],5)
    res += temp
    return res

if __name__ == '__main__':
    data = np.genfromtxt('train.txt')
    L = 60  #len(A)
    num = randint(0, L)
    A = data[num][:-1]
    num = randint(0, L)
    B = data[num][:-1]
    D = float('inf')
    print('LB_Enhanced:'LB_Enhanced(A,B,6,20,D))
import numpy as np

# z-normalized time series;消除序列之间振幅的差异
def Z_normalization(x,mu,sigma):
    x = (x - mu) / sigma
    return x

def SGD(a,v):
    # aligned sequence y' towars x
    CC = np.correlate(a,v,mode = 'full')
    # NCCc;normalization
    CC = CC / np.sqrt(np.dot(a,a)*np.dot(v,v))
    # print(len(CC))
    print(CC)
    value = float('-inf')
    for i in range(len(CC)):
        if CC[i] > value:
            value = CC[i]
            index = i
    # print(index)
    return value,index

def inner_product(a,b):
    product = np.dot(a,b)
    product = product / (np.sqrt(np.dot(a,a)) * np.sqrt(np.dot(b,b)))
    return 1-product # 越小越好

def multi_simlarity(x,y):
    CCw = []
    n = x.shape[0]
    m = x.shape[1]
    for w in range(2*n-1):
        k = w+1-n
        sim = 0
        if k >= 0 :
            for i in range(1,n-k+1):
                temp = inner_product(x[i+k-1,:],y[i-1,:])
                sim = sim + temp
        else:
            k = -k
            for i in range(1,n-k+1):
                temp = inner_product(y[i+k-1,:],x[i-1,:])
                sim = sim + temp
        CCw.append(sim)
    value = float('inf')
    for i in range(len(CCw)):
        if CCw[i] < value:
            value = CCw[i]
            # index = i
    return value

def main():
    a = np.array([1, 1, 1, 1, 2, 4, 4, 3])
    v = np.array([2, 2, 1, 1, 1, 4, 4, 3])
    value,index = SGD(Z_normalization(a,np.mean(a),np.std(a)),Z_normalization(v,np.mean(v),np.std(v)))
    dist = 1 - value # 0-2
    shift = index - len(v)
    if shift >= 0:
        aligned_a = np.concatenate((a[shift:],np.zeros(shift)))
    else:
        aligned_a = np.concatenate((np.zeros(abs(shift)),a[:index+1]))
    print(aligned_a)

if __name__ == '__main__':
    main()
import numpy as np
import fssCPA

def beta_range():
    xx = np.arange(0.05,0.95,0.05)
    train_data,label = fssCPA.concat_data()
    n = train_data.shape[0]
    tDTW_data = np.zeros((n,n),dtype=np.float32) #对角线上的默认为0
    for i in range(n-1):
        for j in range(i+1,n):
            tDTW_data[i,j] = fssCPA.tDTW_calculate(train_data[i],train_data[j])
            tDTW_data[j,i] = tDTW_data[i,j]
    error = []
    for beta in xx:
        aDTW_data = fssCPA.aDTW_calculate(train_data,train_data,beta)
        err = abs(aDTW_data - tDTW_data) / tDTW_data
        for k1 in range(n):
            for k2 in range(n):
                if math.isinf(err[k1,k2]):
                    err[k1,k2] = 0
        error.append(np.sum(err)/(n+n))
    plt.plot(xx,error,'-')

beta_range()
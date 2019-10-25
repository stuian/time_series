import numpy as np

def kernel(x,y):
    n = len(x)
    m = len(y)

    yita = 10
    disMatrix = np.zeros((n,m))
    diffMatrix = np.zeros((n,m))

    for i in range(n):
        for j in range(m):
            disMatrix[i,j] = np.exp(-(x[i]-y[j])**2/yita)


    diffMatrix[0,0] = disMatrix[0,0]
    for i in range(1,n):
        diffMatrix[i,0] = diffMatrix[i-1,0] * disMatrix[i,0]
    for j in range(1,m):
        diffMatrix[0,j] = diffMatrix[0,j-1] * disMatrix[0,j]
    for i in range(1,n):
        for j in range(1,m):
            diffMatrix[i,j] = (diffMatrix[i,j-1] + diffMatrix[i-1,j] + diffMatrix[i-1,j-1]) * disMatrix[i,j]

    return diffMatrix[n-1,m-1]

def update_subspace(in_cluster,out_cluster,center_label,X,single_distance_between,alpha,beta):
    K = len(in_cluster)
    R = X.shape[2]

    # 更新变量子空间
    # 方法二：MHA
    W = np.zeros((K, R))
    MMD = np.zeros((K,R))
    pkr = np.zeros((K,R))


    for k in range(K):
        length_ck = len(in_cluster[k])
        length_nk = len(out_cluster[k])
        if length_ck > 1:
            for r in range(R):
                interdistance = np.zeros((length_ck,length_ck))
                outsidedistance = np.zeros((length_nk ,length_nk))
                betweendistance = np.zeros((length_ck,length_nk))

                for i in range(length_ck):
                    for j in range(i+1,length_ck):
                        interdistance[i,j] = kernel(X[in_cluster[k][i]][:,r],X[in_cluster[k][j]][:,r])
                        interdistance[j,i] = interdistance[i,j]
                interD = np.sum(interdistance) / (length_ck*(length_ck-1))

                for i in range(length_nk):
                    for j in range(i + 1, length_nk):
                        outsidedistance[i, j] = kernel(X[out_cluster[k][i]][:, r], X[out_cluster[k][j]][:, r])
                        outsidedistance[j, i] = outsidedistance[i, j]
                outerD = np.sum(outsidedistance) / (length_nk * (length_nk - 1))

                for i in range(length_ck):
                    for j in range(length_nk):
                        betweendistance[i,j] = kernel(X[in_cluster[k][i]][:, r],X[out_cluster[k][j]][:, r])
                betweenD = np.sum(betweendistance) / (length_ck*length_nk)

                MMD[k,r] = abs(interD + outerD - 2*betweenD)

                for i in in_cluster[k]:
                    pkr[k,r] = pkr[k,r] + single_distance_between[i,center_label[k],r]**2
                pkr[k,r] = np.sqrt(pkr[k,r]) / length_ck # runningtime说明length_ck里出现了0

            MMD[k,:] = 1.0/(np.power(MMD[k,:],1/alpha))
            MMD[k,:] = MMD[k,:] / np.sum(MMD[k,:])
            pkr[k,:] = pkr[k,:] / np.sum(pkr[k,:])
            W[k,:] = pkr[k,:] * MMD[k,:]
            W[k,:] = np.exp(-W[k,:]/beta)
            W[np.isnan(W)] = 1 / R
            sumkr = np.sum(W[k,:])
            if sumkr !=0 :
                W[k, :] = W[k,:] / sumkr
            else:
                W[k, :] = 1 / R
        else:
            W[k, :] = 1 / R
    return W

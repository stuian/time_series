import numpy as np
import load_data

# 数据的10%作为pairwise constraint


def K_means_fDTW_auxi(C,train_data,beta, label,needSigma,useBound, MustLink,CannotLink):
    iter = 15
    k = C.shape[0]
    n = train_data.shape[0]
    if useBound == 0:
        aDTW = aDTW_calculate(C,train_data, beta)
    else:
        aDTW = Bound_Calculate(C, train_data, useBound)

    index = np.zeros(n)

    if needSigma == 1:
        aDTW_std = np.std(aDTW)
        sigma = sigmoid(aDTW_std) * aDTW_std
        sigma = sigma * 2  # δ?
    else:
        sigma = 0
    classes = {}
    for i in range(k):
        classes[i] = [] #？
    for i in range(iter):
        for p in range(n):
            curr_colum = aDTW[:, p]
            temp = map(list(curr_colum).index, heapq.nsmallest(2, curr_colum))
            temp = list(temp)
            index1 = temp[0]
            min1 = curr_colum[index1]
            index2 = temp[1]
            min2 = curr_colum[index2]
            changed = False
            if abs(min2 - min1) >= sigma:
                # add CannotLink
                for s in range(CannotLink.shape[0]):
                    if (p == CannotLink[s,0] and CannotLink[s,1] in classes[index1]) \
                            or (p == CannotLink[s,1] and CannotLink[s,0] in classes[index1]):
                        changed = True
                        break
                if changed:
                    changed = False
                    index[p] = index2
                else:
                    index[p] = index1
            else:
                min1 = tDTW_calculate(C[index1], train_data[p])
                min2 = tDTW_calculate(C[index2], train_data[p])
                aDTW[index1, p] = min1
                aDTW[index2, p] = min2
                if min1 < min2:
                    for s in range(CannotLink.shape[0]):
                        if (p == CannotLink[s, 0] and CannotLink[s, 1] in classes[index1]) \
                                or (p == CannotLink[s, 1] and CannotLink[s, 0] in classes[index1]):
                            changed = True
                            break
                    if changed:
                        changed = False
                        index[p] = index2
                    else:
                        index[p] = index1
                else:
                    for s in range(CannotLink.shape[0]):
                        if (p == CannotLink[s, 0] and CannotLink[s, 1] in classes[index2]) \
                                or (p == CannotLink[s, 1] and CannotLink[s, 0] in classes[index2]):
                            changed = True
                            break
                    if changed:
                        changed = False
                        index[p] = index1
                    else:
                        index[p] = index2
            classes[index[p]].append(p)
            # add MustLink
            for u in range(MustLink.shape[0]):
                if p == MustLink[u,0]:
                    index[MustLink[u,1]] = index[p]
                    classes[index[p]].append(MustLink[u,1])
                elif p == MustLink[u,1]:
                    index[MustLink[u,0]] = index[p]
                    classes[index[p]].append(MustLink[u,0])

        # update Center point
        for k1 in range(k):
            clust_ind = []
            for ind, j in enumerate(index):
                if j == k1:
                    clust_ind.append(ind)
            temp = np.zeros(C.shape[1])
            for k2 in range(clust_index):
                temp = temp + train_data[k2]
            temp = temp / len(clust_ind)
            C[k1] = temp  # 更新
        aDTW = aDTW_calculate(C, train_data, beta)
    return index,RandIndex(index,label)

if __name__ == '__main__':
    X,y = load_data.concat_data()
    num = X.shape[0]
    rand_arr = np.arange(num)
    np.random.shuffle(rand_arr)
    X_CPA = X[rand_arr[:int(num*0.1)]]
    # 生成center集

    # mustlink和cannotlink
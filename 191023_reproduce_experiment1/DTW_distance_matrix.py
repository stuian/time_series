import numpy as np

def DTW(s1,s2):
    DTW = {}
    for i in range(len(s1)):
        DTW[(i, -1)] = float('inf')
    for i in range(len(s2)):
        DTW[(-1, i)] = float('inf')
    DTW[(-1, -1)] = 0
    for i in range(len(s1)):
        for j in range(len(s2)):
            dist = (s1[i] - s2[j]) ** 2
            DTW[(i, j)] = dist + min(DTW[(i - 1, j)], DTW[(i, j - 1)], DTW[(i - 1, j - 1)])
    return np.sqrt(DTW[(len(s1) - 1, len(s2) - 1)])

def distance_matrix(X):
    n = X.shape[0]
    R = X.shape[2]
    single_distance_between = np.zeros((n,n,R))
    for i in range(1,n):
        for j in range(i):
            for r in range(R):
                single_distance_between[i,j,r] = DTW(X[i][:,r],X[j][:,r])
                single_distance_between[j,i,r] = single_distance_between[i,j,r]
    return single_distance_between
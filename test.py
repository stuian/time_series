import numpy as np

def init_subspace_weight(r,k):
    # r是变量的个数，k是聚类中心的个数
    variable_weight = globals()
    for i in range(k):
        variable_weight['s%s' % k+1] = [] # s1,...,sk
        for j in range(r):
            variable_weight['s%s' % k+1].append(1/r)

init_subspace_weight(2,3)
print(variabel_weight)
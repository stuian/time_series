import numpy as np
a = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(a.shape[0])
rand_arr = np.arange(a.shape[0])
print(rand_arr)
print(np.random.shuffle(rand_arr))
print(a[rand_arr])
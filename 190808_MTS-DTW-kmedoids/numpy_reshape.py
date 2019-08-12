import numpy as np


aa = np.array([[[1,2,3],[4,5,6]],[[2,2,3],[4,5,6]],[[3,2,3],[4,5,6]]])
a = np.array([[4,2,3],[4,5,6]])
print(aa.shape)
print(a.shape)
data = np.append(aa,a) # 先拼成一个行向量
print(data)
print(data.shape)

data1 = data.reshape(aa.shape[0]+1,aa.shape[1],aa.shape[2])
print(data1.shape)

# -------------------------

b = np.arange(36).reshape((6,6))
b1 = b.reshape(2,3,6)
print(b)
print(b1) # 把行拆了
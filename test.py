import numpy as np

np.random.seed(0)
x = np.random.random((3,3))
print(x)
s = x.sum(axis=1)
for i in range(3):
    x[i] = x[i] / s[i]
print(x)

print(x.sum(axis=1))
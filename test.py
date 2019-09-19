import numpy as np

x = np.random.uniform(-10,10,(100, 2))
print(np.sum(np.square(x), axis=1))
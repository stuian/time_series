import numpy as np

file = 'distance'
PATH = './190915_subspace-clustering/data/' + file + '.npy'
data = np.load(PATH)
print(data.shape)

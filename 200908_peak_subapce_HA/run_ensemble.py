import numpy as np
import os
from sklearn.cluster import AffinityPropagation
from evaluation import RandIndex
from evaluation import Purity
from evaluation import NMI
import h5py
from two_level_consistency import two_level_consistency

def center_value(matrix):
    x = []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            x.append(matrix[i,j])
    length = len(x)
    x.sort()
    if length % 2 == 1:
        z = length // 2
        y = x[z]
    else:
        y = (x[length//2]+x[length//2-1])/2
    return y

file = "lp5"
path = './data/'
filename = file + '.h5'
filename = os.path.join(path, filename)
f = h5py.File(filename, 'r')
X = f['train_x'][:]
y = f['train_y'][:]
PATH = './result/'
K = len(np.unique(y))
N = len(y)

parts_name = file + '_parts.npy'
parts_path = os.path.join(PATH,parts_name)
parts = np.load(parts_path)

cluster_contribute = two_level_consistency(parts, 0.8)
W = np.diag(np.array(cluster_contribute))
B = 50
for b in range(B):
    part = parts[b]
    if b == 0:
        H = np.zeros((N, K))
        for n in range(N):
            H[n, int(part[n])] = 1
    else:
        temp = np.zeros((N, K))
        for n in range(N):
            temp[n, int(part[n])] = 1
        H = np.hstack((H, temp))
S = np.dot(H.dot(W), H.T) / B

concensus_name = file + '_concensus.npy'
concensus_path = os.path.join(PATH, concensus_name)
concensus_matrix = np.load(concensus_path)

S_name = file + '_S.npy'
S_path = os.path.join(PATH, S_name)
S_no_weight = np.load(S_path)

centers = center_value(concensus_matrix)
af = AffinityPropagation(affinity='precomputed', damping=0.9, preference=0.9).fit(concensus_matrix)
labels = af.labels_

center = center_value(S)
af = AffinityPropagation(affinity='precomputed', damping=0.9, preference=0.9).fit(S)
y_pred = af.labels_

center_no_weight = center_value(S_no_weight)
af = AffinityPropagation(affinity='precomputed', damping=0.9, preference=0.9).fit(S_no_weight)
y_pred_no_weight = af.labels_


RI_value = RandIndex(labels, y)
purity_value = Purity(labels, y)
NMI_value = NMI(labels, y)
print(RI_value,purity_value,NMI_value)

RI_value = RandIndex(y_pred, y)
purity_value = Purity(y_pred, y)
NMI_value = NMI(y_pred, y)
print(RI_value,purity_value,NMI_value)

RI_value = RandIndex(y_pred_no_weight, y)
purity_value = Purity(y_pred_no_weight, y)
NMI_value = NMI(y_pred_no_weight, y)
print(RI_value,purity_value,NMI_value)
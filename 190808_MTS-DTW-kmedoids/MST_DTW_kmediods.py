import numpy as np
import random
import math
import h5py
import time

def cluster(distances, k):
    # Pick k random medoids.
    curr_medoids = np.array([-1] * k)
    while len(np.unique(curr_medoids)) != k:
        curr_medoids = np.array([random.randint(0, num) for _ in range(k)])
    old_medoids = np.array([-1]*k)
    new_medoids = np.array([-1]*k)

    # To be repeated until mediods stop updating
    while not ((old_medoids == curr_medoids).all()):
        # Assign each point to cluster with closest medoid.
        clusters = assign_points_to_clusters(curr_medoids, distances)
        # Update cluster medoids to be lowest cost point.
        for curr_medoid in curr_medoids:
            cluster = np.where(clusters == curr_medoid)[0]
            new_medoids[curr_medoids == curr_medoid] = compute_new_medoid(cluster, distances)

        old_medoids[:] = curr_medoids[:]
        curr_medoids[:] = new_medoids[:]
        print('Mediods still not equal')

def assign_points_to_clusters(medoids, distances):
    distances_to_medoids = distances[:,medoids]
    clusters = medoids[np.argmin(distances_to_medoids, axis=1)] # numpy.argmin表示最小值在数组中所在的位置
    clusters[medoids] = medoids
    return clusters

def DTWDistance(s1,s2,w):
    DTW={}
    w = max(w, abs(len(s1)-len(s2)))

    for i in range(-1,len(s1)):
        for j in range(-1,len(s2)):
            DTW[(i, j)] = float('inf')
    DTW[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(max(0, i-w), min(len(s2), i+w)):
            dist= (s1[i]-s2[j])**2
            DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])

    return np.sqrt(DTW[len(s1)-1, len(s2)-1])

if __name__ == '__main__':
    # 导入数据
    file = h5py.File('cricket_data.h5', 'r')
    Train_data = file['train_x'][:]
    train_y = file['train_y'][:]
    file.close()

    # Calculate distances using DTW
    num = np.shape(Train_data)[0]
    features = np.shape(Train_data)[1]
    length = np.shape(Train_data)[2]
    distances = np.zeros((num,num))
    # window size
    w = np.shape(Train_data)[2] // 10 # 长度的百分之十
    start = time.time()
    for i in range(num-1):
        for j in range(i+1,num): # 上三角
            cur_dist = 0.0
            for z in range(features):
                cur_dist += DTWDistance(Train_data[i][z,:],Train_data[j][z,:],w)
            distances[i, j] = cur_dist
            distances[j, i] = distances[i, j]
    end = time.time()
    print('calculating distances has spent %ds' % int(end-start))
    dist = h5py.File("distances.h5",'w')
    dist.create_dataset('distances',data=distances)
    dist.close()
    print('Distances have already calculated')

    clusters, curr_medoids = cluster(distances, 3)

    # new_medoids[medoids == medoid] = sample # 其实是当medoids取medoid，把medoid用sample替换了
    



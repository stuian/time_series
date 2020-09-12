import numpy as np

def series_to_centers(single_distance_between, x, center_label, W):
    minDist = float('inf')
    index = -1
    R = W.shape[1]
    for i in range(len(center_label)):
        if x == center_label[i]:
            return i
        else:
            temp = 0
            for r in range(R):
                # print(center_label[i])
                # print(single_distance_between[x,int(center_label[i]),r])
                # print(W[i,r])
                temp += np.power(single_distance_between[x,int(center_label[i]),r],2) * W[i,r]
            temp = np.sqrt(temp)
            if temp < minDist:
                minDist = temp
                index = i
    return index
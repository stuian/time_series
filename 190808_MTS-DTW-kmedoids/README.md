# DTW_kmedoids
Multivariate time series clustering using Dynamic Time Warping (DTW) and k-mediods algorithm
This repository contains code for clustering of multivariate time series using DTW and k-mediods algorithm. It contains code for optional use of LB_Keogh method for large data sets that reduces to linear complexity compared to quadratic complexity of dtw.
The train data should be a numpy array of the form (M,N,D) where
1. M - Number of data sequences.
2. N - length of data sequences.
3. D - Dimension of data sequences (number of features).

The algorithm was tested on a synthetic vehicle encounter dataset. Clustering vehicle encounter data into different kinds of encounters - 
The dataset contained time sequences of 100 steps (duration of 10s) belonging to 3 different classes. Each class had 200 samples.
![alt text](https://github.com/aditya1709/DTW_kmedoids/blob/master/Confusion_matrix_c.png)

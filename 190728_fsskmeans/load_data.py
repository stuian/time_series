import os
import numpy as np

def load_data(path):
    """
    :param path:所读取文件的路径
    :return:
    """
    file_list = os.listdir(path)
    count = 0
    for file in file_list:  # 127个
        final_path = os.path.join(path, file)
        if count == 0:
            data = np.genfromtxt(final_path, delimiter=' ')
        else:
            temp = np.genfromtxt(final_path, delimiter=' ')
            data = np.vstack((data, temp))
        count += 1
    return data

def concat_data():
    """
    wafer dataset
    :return: X,y
    """
    abnormal_path = "./data/wafer/abnormal"
    normal_path = "./data/wafer/normal"
    abnormal_data = load_data(abnormal_path)
    abnormal_data = np.c_[abnormal_data,np.ones(len(abnormal_data))]

    normal_data = load_data(normal_path)
    normal_data = np.c_[normal_data,np.zeros(len(normal_data))]

    data = np.vstack((abnormal_data,normal_data))
    # print(data.shape) #(2756, 7)
    # print(data[0])    #[  2. -11.  -1.   3.  24.  10.   1.]
    return data[:,:-1],data[:,-1]

if __name__ == '__main__':
    X,y = concat_data()
    print(X[0])
    print(y[0])
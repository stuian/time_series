import numpy as np
import h5py
import os

# 正规化数据集

# 1、
def cricket_data():
    xleft = np.genfromtxt('../data/Cricket/xleft.txt',delimiter=' ')
    label = xleft[:,0]
    xleft = xleft[:,1:]
    yleft = np.genfromtxt('../data/Cricket/yleft.txt',delimiter=' ')[:,1:]
    zleft = np.genfromtxt('../data/Cricket/zleft.txt',delimiter=' ')[:,1:]
    xright = np.genfromtxt('../data/Cricket/xright.txt',delimiter=' ')[:,1:]
    yright = np.genfromtxt('../data/Cricket/yright.txt',delimiter=' ')[:,1:]
    zright = np.genfromtxt('../data/Cricket/zright.txt',delimiter=' ')[:,1:]
    # print(xleft.shape)
    # print(yleft.shape)
    # print(zleft.shape)
    # print(xright.shape)
    # print(yright.shape)
    # print(zright.shape) # (180, 1198-1)

    m = xleft.shape[0]
    n = xleft.shape[1]
    num = 6
    data = np.concatenate((xleft,yleft,zleft,xright,yright,zright),axis=1)
    data = data.reshape(m,num,n) # 样本数量；变量数目；样本长度
    data = data.transpose(0,2,1)
    # print(data.shape)  # (180, 1197, 6)

    file = h5py.File('cricket_data.h5','w')
    file.create_dataset('train_x',data = data)
    file.create_dataset('train_y',data = label)
    file.close()

def train_test_split():
    pass

# 3、
def get_filelist(dir):
    Dirlist = []
    labelname = []
    for s in os.listdir(dir):
        labelname.append(s)
        newDir = os.path.join(dir,s)
        Dirlist.append(newDir)
    labels = []
    count = 0
    for ind,i in enumerate(Dirlist): # 样本长度不一样
        for file in os.listdir(i):
            newpath = os.path.join(i,file)
            print(newpath)
            if count == 0:
                temp = np.genfromtxt(newpath,delimiter=' ')
                features = temp.shape[1]
                print('length:%s,features:%s' % (temp.shape[0],temp.shape[1]))
            else:
                temp = np.vstack((temp,np.genfromtxt(newpath,delimiter=' ')))
            labels.append(ind+1) # 1,2,3,4
            count += 1
    data = temp.reshape(count,-1,features)
    print(data.shape,len(labels),labels[0])
    return data,np.array(labels),np.array(labelname)

def full_data(data_name):
    path = 'E:\\Jade\\time_series\\data\\multi_var'
    for file in data_name:
        newpath = os.path.join(path, file)
        data,label,labelname = get_filelist(newpath)
        filename = file + '.h5'
        file = h5py.File(filename, 'w')
        file.create_dataset('train_x', data=data)
        file.create_dataset('train_y', data=label)

        dt = h5py.special_dtype(vlen=str)
        ds = file.create_dataset('labels',labelname.shape,dtype=dt) # labelname是list
        ds[:] = labelname
        file.close()


if __name__ == '__main__':
    # 1、read cricket_data
    # file = h5py.File('cricket_data.h5','r')
    # train_x = file['train_x'][:]
    # train_y = file['train_y'][:]
    # print(len(np.unique(train_y)))
    # 2、Multi 数据
    # data_name = ['Robot Execution Failures lp1', 'Robot Execution Failures lp2', 'Robot Execution Failures lp3','Robot Execution Failures lp4', 'Robot Execution Failures lp5']
    data_name = ['wafer']
    full_data(data_name) # 可能每个样本shape不完全一样，不能reshape
    # read



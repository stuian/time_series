import numpy as np
import math

def Purity(y_pred,labels):
    """
    :param train_labels:y_pred
    :param labels:
    :return:
    """
    length = len(labels)
    classIndex = np.unique(y_pred)
    classNum = len(classIndex)
    maxSum = 0
    for i in range(classNum):
        currIndex = classIndex[i]
        currDataIndex = []
        for j in range(length):
            if y_pred[j] == currIndex:
                currDataIndex.append(j)
        for k in currDataIndex:
            if y_pred[k] == labels[k]:
                maxSum += 1
    return maxSum/length

def RandIndex(y_pred,labels):
    """
    :param y_pred:predict label
    :param labels: true label
    :return:
    """
    length = len(labels)
    TP,TN ,FP,FN = 0,0,0,0
    for k1 in range(length-1):
        for k2 in range(k1+1,length):
            if y_pred[k1]== y_pred[k2] and labels[k1]==labels[k2]:
                # 本身是同一类，且被分到了同一类
                TP = TP + 1
            elif y_pred[k1] != y_pred[k2] and labels[k1]!=labels[k2]:
                # 本身不是同一类，且被分到了不同类
                TN = TN +1
            elif y_pred[k1] == y_pred[k2] and labels[k1] != labels[k2]:
                # 不同类被分到同一类
                FP = FP +1
            elif y_pred[k1] != y_pred[k2] and labels[k1] == labels[k2]:
                # 同一类被分到不同类
                FN = FN +1
    return (TP+TN)/(TP+FP+FN+TN)

def NMI(A,B):
    """
    https://blog.csdn.net/chengwenyao18/article/details/45913891
    :param A:np.array
    :param B:np.array
    :return:
    """
    #样本点数
    total = len(A)
    A_ids = set(A)
    B_ids = set(B)
    #互信息计算
    MI = 0
    eps = 1.4e-45
    for idA in A_ids:
        for idB in B_ids:
            idAOccur = np.where(A==idA)
            idBOccur = np.where(B==idB)
            idABOccur = np.intersect1d(idAOccur,idBOccur) #取两者公共包含的数
            px = 1.0*len(idAOccur[0])/total
            py = 1.0*len(idBOccur[0])/total
            pxy = 1.0*len(idABOccur)/total
            MI = MI + pxy*math.log(pxy/(px*py)+eps,2) #为什么要加eps
    # 标准化互信息
    Hx = 0
    for idA in A_ids:
        idAOccurCount = 1.0*len(np.where(A==idA)[0])
        Hx = Hx - (idAOccurCount/total)*math.log(idAOccurCount/total+eps,2)
    Hy = 0
    for idB in B_ids:
        idBOccurCount = 1.0*len(np.where(B==idB)[0])
        Hy = Hy - (idBOccurCount/total)*math.log(idBOccurCount/total+eps,2)
    MIhat = 2.0*MI/(Hx+Hy)
    return MIhat

if __name__ == '__main__':
    pass
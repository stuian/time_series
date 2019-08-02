"""
date: 20190730
theme:层次聚类
"""

from queue import PriorityQueue
import math
import codecs
import numpy as np
from sklearn.cluster import AgglomerativeClustering


# 法一：调用sklearn包
def AggloCluster():
    X = [[1,2],[3,2],[4,4],[1,2],[1,3]]
    clustering = AgglomerativeClustering().fit(x)
    """
    array([1, 0, 0, 1, 1]) # clustering.labels_
    array([[0, 3],
       [4, 5],
       [1, 2],
       [6, 7]]) # clustering.children_
    """

# 法二:
class HCluster(object):
    def getMedian(self,alist):
        # 求一列的中位数
        tmp = list(alist)
        tmp.sort()
        alen = len(tmp)
        if alen % 2 == 1:
            return tmp[alen // 2]
        else:
            return (tmp[alen // 2] + tmp[(alen // 2) -1]) / 2
    def normalize(self,column):
        # 对数值型数据进行归一化，使用绝对标准分
        median = self.getMedian(column)
        asd = sum([abs(x-median) for x in column]) / len(column) # 绝对标准差
        result = [(x-median) / asd for x in column]
        return result

    def distance(self,i,j):
        # 欧氏距离
        sumSquares = 0
        for k in range(1,self.cols):
            sumSquares += (self.data[k][i] - self.data[k][j])**2
        return math.sqrt(sumSquares)

    def __init__(self,filepath):
        self.data = {}
        self.counter = 0
        self.queue = PriorityQueue()
        line_1 = True
        with codecs.open(filepath,'r','utf-8') as f:
            for line in f:
                if line_1:
                    line_1 = False
                    header = line.split(',')
                    self.cols = len(header)
                    self.data = [[] for i in range(self.cols)]
                else:
                    instances = line.split(',')
                    toggle = 0
                    for instance in range(self.cols):
                        if toggle == 0:
                            self.data[instance].append(instances[instance])
                            toggle = 1
                        else:
                            self.data[instance].append(float(instances[instance]))
        for i in range(1,self.cols):
            self.data[i] = self.normalize(self.data[i])
        rows = len(self.data[0])
        for i in range(rows):
            minDistance = np.float('inf')
            nearestNeighbor = 0
            neighbors = {}
            for j in range(rows):
                if i !=j:
                    dist = self.distance(i,j)
                    if i<j:
                        pair = (i,j)
                    else:
                        pair = (j,i)
                    # 计算元素i到所有其它元素的距离，放到邻居字典中，比如i=1,j=2...，结构如i=1的邻居-》{2: ((1,2), 1.23),  3: ((1, 3), 2.3)... }
                    neighbors[j] = (pair,dist)
                    if dist < minDistance:
                        minDistance = dist
                        nearestNeighbor = j
            # 创建最邻近对
            if i < nearestNeighbor:
                nearesPair = (i,nearestNeighbor)
            else:
                nearesPair = (nearestNeighbor,i)
            # 放入优先对列中，(最近邻距离，counter,[label标签名，最近邻元组，所有邻居])
            self.queue.put((minDistance,self.counter,[[self.data[0][i]],nearestPair,neighbors]))
            self.counter += 1

    def cluster(self):
        done = False
        while not done:
            topOne = self.queue.get()
            nearestPair = topOne[2][1]
            if not self.queue.empty():
                nextOne = self.queue.get()
                nearPair = nextOne[2][1]
                tmp = []
                while nearPair != nearestPair:
                    tmp.append((nextOne[0],self.counter,nextOne[2]))
                    self.counter += 1 # 从__init__结束后的counter值开始
                    nextOne = self.queue.get()
                    nearPair = nextOne[2][1]
                # 重新加回Pop出的不相等最近邻的元素
                for item in tmp:
                    self.queue.put(item)
                if len(topOne[2][0]) == 1: # 多标签？
                    item1 = topOne[2][0][0]
                else:
                    item1 = topOne[2][0]
                if len(nextOne[2][0]) == 1:
                    item2 = nextOne[2][0][0]
                else:
                    item2 = nextOne[2][0]
                # #联合两个最近邻族成一个新族
                curCluster = (item1,item2)

                minDistance = np.float('inf')
                nearestPair = ()
                nearestNeighbor = ''
                merged = {}
                nNeighbors = nextOne[2][2]
                for key,value in topOne[2][2].items():
                    if key in nNeighbors:
                        if nNeighbors[key][1] < value[1]:
                            dist = nNeighbors[key]
                        else:
                            dist = value
                        if dist[1] < minDistance:
                            minDistance = dist[1]
                            nearestPair = dist[0]
                            nearestNeighbor = key
                        merged[key] = dist
                if merged == {}:
                    return curCluster
                else:
                    self.queue.put((minDistance,self.counter,[curCluster,nearestPair,merged]))

if __name__=='__main__':
    hcluser=HCluster('filePath')
    cluser=hcluser.cluster()
    print(cluser)


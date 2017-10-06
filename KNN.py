# -*- coding: utf-8 -*-
'''k近邻算法'''

import numpy as np
from collections import Counter

# data:数据，d:预测数据，k:近邻数
def KNN(data, d, k):
    x = data[:,:-1].astype(np.float64)
    y = data[:,-1]
    mdis = np.sqrt(np.sum((x-d)**2, axis=1))
    mdi = np.argsort(mdis)
    yk = Counter([y[mdi[i]] for i in range(k)])
    return sorted(yk,key=lambda x:yk[x])[-1]

# 读取数据
def readData():
    data = np.array([[0,0,'a'], [1,1,'a'], [0,1,'a'], [4,1,'a'],
                     [4,3,'b'], [5,5,'b'], [3,4,'b'], [1,0,'b'],
                     [-1,-1,'c'], [-2,0,'c'], [0,-2,'c']])
    return data

if __name__ == '__main__':
    data = readData()
    pp = np.array([0,0])
    print(KNN(data, pp, 3))
    
    
    
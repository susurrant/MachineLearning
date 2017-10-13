# -*- coding: utf-8 -*-
'''贝叶斯分类（贝叶斯估计）'''

import numpy as np
from collections import Counter

def bayes(data, x, lmd=1):
    yCounter = Counter(data[:,-1])
    yList = list(yCounter.keys())
    cp_x_y = np.zeros((len(x), len(yList)))

    for i in range(len(x)):
        Sl = len(set(data[:,i]))*lmd
        td = data[np.where(data[:,i]==str(x[i]))]
        for j,y in enumerate(yList):
            cp_x_y[i, j] = (len(td[np.where(td[:,-1]==y)])+lmd)/(yCounter[y]+Sl)

    prob = np.row_stack((cp_x_y, np.array([(yCounter[y]+lmd)/(data.shape[0]+len(yList)*lmd) for y in yList])))
    return yList[np.argmax(np.prod(prob, axis=0))]

def readData():
    data = np.array([[1,'S',-1], [1,'M',-1], [1,'M',1], [1,'S',1], [1,'S',-1],
                     [2,'S',-1], [2,'M',-1], [2,'M',1], [2,'L',1], [2,'L',1],
                     [3,'L',1], [3,'M',1], [3,'M',1], [3,'L',1], [3,'L',-1]])
    return data

if __name__ == '__main__':
    data = readData()
    print(bayes(data, [2,'S']))
# -*- coding: utf-8 -*-
'''普通K均值聚类算法'''

import numpy as np

def initCenters(data, k, method = 1):
    rdidx = []
    
    if method == 0: # 完全随机质心选取，算法不稳定
        while len(set(rdidx)) != k:
            rdidx = np.random.random_integers(0, data.shape[0]-1, k)
    elif method == 1: # 最远距离选取，适用于无或很少噪声的数据
        rdidx.append(np.random.randint(data.shape[0]))
        c = []
        c.append(data[rdidx[0]])
        data = np.delete(data, rdidx[0], axis=0)
        for i in range(k-1):
            totalDis = np.zeros((data.shape[0],))
            for tc in c:
                totalDis += np.sqrt(np.sum((data-tc)**2, axis=1))
            rdidx.append(np.argmax(totalDis))
            c.append(data[rdidx[-1]])
            data = np.delete(data, rdidx[-1], axis=0)
    
    return rdidx

def KMeans(data, k):
    rdidx = initCenters(data[:], k)
    c = data[rdidx]
    l = [0]*data.shape[0]

    while True:
        tag = False
        for i,d in enumerate(data):
            idx = np.argmin(np.sum((d-c)**2, axis=1))
            if l[i] != idx:
                l[i] = idx
                tag = True
        
        if tag:
            for i in range(k):
                idx = np.where(np.array(l) == i)
                if idx:
                    c[i] = np.mean(data[idx], axis=0)
        else:
            break
    
    return l


def readData():
    data = np.array([[0,0], [0,1], [1,0], [1,1], [-1,0],
                     [99,99], [100,100], [99,100], [100,99],
                     [0,99], [1,99], [1,100], [0,100]])
    return data

if __name__ == '__main__':
    data = readData()
    print(KMeans(data, 3))
    
# -*- coding: utf-8 -*-
'''决策树(C4.5)'''

import numpy as np
from collections import Counter

def infoGain(data, i, HD):
    vCounter = Counter(data[:, i])
    H_DA = 0
    subData = {}
    for vk in vCounter:
        td = data[np.where(data[:, i] == str(vk))]
        labelP = np.array(Counter(td[:, -1]).values())/td.shape[0]
        H_DA -= td.shape[0]*np.sum(labelP*np.log2(labelP))/data.shape[0]
        subData[vk] = td
    H_AD = np.array(vCounter.values())/data.shape[0]
    H_AD = -1*np.sum(H_AD*np.log2(H_AD))

    return (HD-H_DA)/H_AD, subData


def subTree(data, A, eps):
    if len(set(data[:,-1])) == 1:
        return {'tree': data[0,-1]}
    if not len(A):
        c = Counter(data[:, -1])
        a = max(c, key=c.get)
        return {'tree': a}




def chooseBestFeature(data, A, eps):
    ig = 0
    HD = 0
    for i in A:
        tem_ig, subData = infoGain(data, i, HD)
        if tem_ig > ig:
            Ag = i
            ig = tem_ig

    if ig < eps:
        c = Counter(data[:, -1])
        a = max(c, key=c.get)
        return {'tree': a}


def decisionTree(data, A, eps):
    if len(set(data[:,-1])) == 1:
        return {'tree': data[0,-1]}

    if not len(A):
        c = Counter(data[:, -1])
        a = max(c, key=c.get)
        return {'tree': a}


    a, subData = chooseBestFeature(data, A, eps)
    for k,v in subData.items():

        A.remove(a)

if __name__ == '__main__':
    #data = readData()
    #print(bayes(data, [2,'S']))
    c = Counter([1,1,2,3,4,5,5,5])
    a = max(c, key=c.get)
    print(a)
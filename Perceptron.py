
import numpy as  np

def train(X, y, step):
    dnum, featureSize = X.shape
    W= np.zeros(featureSize)
    b = 0
    tag = True

    while tag:
        for i in range(dnum):
            if y[i] * (np.dot(X[i], W) + b) <= 0:
                W += step * y[i] * X[i]
                b += step * y[i]
                break
            if i == dnum-1:
                tag = False
    
    return W, b

def test(X, y, W, b):
    for i, x in enumerate(X):
        print(y[i]*(np.dot(x, W)+b))

def readData():
    X = np.array([[3,3], [4,3], [1,1]])
    y = [1,1,-1]
    return X, y

if __name__ == '__main__':
    step = 1
    X, y = readData()
    W, b = train(X, y, step)
    print(W, b)
    test(X, y, W, b)
import numpy as np
from matplotlib import pyplot as plt
data = np.loadtxt("ex1data2.txt", delimiter = ",", dtype = 'int')
x = data[:, 0:2]
y = data[:, 2]
m = len(y)

x = x.reshape(m,2)
y = y.reshape(m,1)
one = np.ones((m,1))

X = np.matrix(x)

X = (X - np.mean(X, axis = 0))/np.std(X, axis = 0)

X = np.concatenate((one,X), axis = 1)

theta = np.zeros((3, 1))

def cc(theta, X, y):
    A = np.dot(X,theta)-y
    return float(((1/(2*m)) * np.dot(A.T, A)))


def gd(theta, X, y, alpha, iterations):
    for i in range(iterations):
        h = np.dot(X, theta)-y
        h = np.dot(X.T, h)
        theta = theta - (alpha/m) * h
    return theta

theta = (gd(theta, X, y, 0.01, 400))

print(cc(theta, X, y))

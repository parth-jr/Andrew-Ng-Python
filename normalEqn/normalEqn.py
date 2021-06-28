import numpy as np
from matplotlib import pyplot as plt

data = np.loadtxt("ex1data2.txt", delimiter = ",")

x = data[:, 0:2]
y = data[:,2]

m = len(y)

x = x.reshape(m,2)
y = y.reshape(m,1)
one = np.ones((m,1))

X = np.concatenate((one,x), axis = 1)
theta = np.zeros((3, 1))

X = (X-np.mean(X))/np.std(X)

def cc(theta, X, y):
    A = np.dot(X,theta)-y
    return float(((1/(2*m)) * np.dot(A.T, A)))

print(cc(theta, X, y))

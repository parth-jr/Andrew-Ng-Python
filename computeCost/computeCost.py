import numpy as np
from matplotlib import pyplot as plt

data = np.loadtxt("ex1data1.txt", delimiter = ",")

x = data[:,0]
y = data[:,1]

m = len(y)

x = x.reshape(m,1)
y = y.reshape(m,1)
one = np.ones((m,1))

X = np.concatenate((one,x), axis = 1)

theta = np.zeros((2, 1))

def cc(theta, X, y):
    A = np.dot(X,theta)-y
    return ((1/(2*m)) * np.dot(A.T, A))

print (cc(theta, X, y))

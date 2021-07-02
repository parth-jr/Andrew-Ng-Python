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

X = (X - np.mean(X, axis = 0))/np.std(X, axis = 0) #axis = 0 means column & axis = 1 means row 

X = np.concatenate((one,X), axis = 1)

print(X)

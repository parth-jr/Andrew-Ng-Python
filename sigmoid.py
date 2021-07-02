import numpy as np
from matplotlib import pyplot as plt

data = np.loadtxt("ex2data1.txt", delimiter = ',', dtype = 'float')

X = data[:, 0:2]
y = data[:, 2]
m = len(y)

X = X.reshape((m,2))
y = y.reshape((m,1))

def sigmoid(x):
      return 1/(1+np.exp(-x))

a = input()
a = int(a)

print(sigmoid(a))


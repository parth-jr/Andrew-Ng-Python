import numpy as np
from matplotlib import pyplot as plt

data = np.loadtxt("ex2data1.txt", delimiter = ',', dtype = 'float')

X = data[:, 0:2]
y = data[:, 2]
m = len(y)

X = X.reshape((m,2))
y = y.reshape((m,1))


for i in range(m):
        if(y[i] == 0):
            plt.plot(X[i,0],X[i,1], "o", color = '#FFDD3C', markersize = 10) #can also use plt.scatter instead
        else:
            plt.plot(X[i,0],X[i,1],"+", color = 'black', markersize = 10)

plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.show()

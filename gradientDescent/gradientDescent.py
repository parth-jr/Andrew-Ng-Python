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
    h = np.dot(X,theta)-y
    return ((1/(2*m)) * np.dot(h.T, h))

def gd(theta, X, y, alpha, iterations):
    for i in range(iterations):
        h = np.dot(X,theta)-y
        h = np.dot(X.T, h)
        theta = theta - (alpha/m) * h
    return theta


print ("Cost Before Gradient Descent:", cc(theta, X, y))
print ("Theta After Gradient Descent:", gd(theta, X, y, 0.01, 1500))

theta = gd(theta, X, y, 0.01, 1500)

print ("Cost After Gradient Descent:", cc(theta, X, y))

graph = input("Do you want a plot for Gradient Descent? (Press Y for Yes & N for No)\n")

if(graph == 'Y' or 'y'):
    plt.plot(x, y, "rx")
    plt.plot(x, np.dot(X,theta))
    plt.show()
else:
    exit()

import numpy as np
from matplotlib import pyplot as plt

data = np.loadtxt("ex2data1.txt", delimiter = ',', dtype = 'float128')

X = data[:, 0:2]
y = data[:, 2]
m = len(y)

X = X.reshape((m,2))
y = y.reshape((m,1))
ones = np.ones((m,1))

theta = np.zeros((X.shape[1]+1,1))

def sigmoid(x):
      return 1/(1+np.exp(-x))

def costFunction(X, y, theta, m):
    hypothesis = sigmoid(np.dot(X, theta))
    J = (-1/m)*np.sum(y*(np.log(hypothesis)) + ((1-y)*np.log(1 - hypothesis)))
    ogradient = (1/m)*(np.dot(X.T, hypothesis - y))
    return J, ogradient

def gradientDescent(theta, X, y, alpha, iterations, m):
    J_plot =[]
    for i in range(iterations):
        J, ogrd = costFunction(X, y, theta, m)
        theta = theta - (alpha * ogrd)
        J_plot.append(J)
    return theta, J_plot

def featureNormalization(x):
    x = (x - np.mean(x, axis = 0))/np.std(x, axis = 0)
    return x

def DecisionBoundary(X, y, theta):
    x_p = (np.min(X[:,1]), np.max(X[:,2]))
    y_p = -(theta[0] + theta[1]*x_p)/theta[2]
    plt.plot(x_p, y_p)

def plotXy(X, y):
    for i in range(m):
            if(y[i] == 0):
                plt.plot(X[i,1],X[i,2], "o", color = '#FFDD3C', markersize = 5) #can also use plt.scatter instead
            else:
                plt.plot(X[i,1],X[i,2],"+", color = 'black', markersize = 5)

X = featureNormalization(X)
X = np.concatenate((ones, X), axis = 1)

print("Pre-Gradient Descent Value:", costFunction(X, y, theta, m)[0])

theta_F, Jplot = (gradientDescent(theta, X, y, 1, 400, m))

print("Theta =\n", theta_F)
print("Post-Gradient Descent Value:", costFunction(X, y, theta_F, m)[0])

plotXy(X, y)
DecisionBoundary(X, y, theta_F)

plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.show()

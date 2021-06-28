import numpy as np
from matplotlib import pyplot as plt

data = np.loadtxt("ex1data1.txt", delimiter = ",")

x = np.array(data[:,0])
y = np.array(data[:,1])

plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.plot(x, y, "rx")
plt.show()

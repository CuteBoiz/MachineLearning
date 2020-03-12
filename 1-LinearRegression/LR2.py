import numpy as np 
import matplotlib.pyplot as plt 

X = np.array([[50, 55, 60, 65, 70, 75, 80]]).T
one = np.ones((7, 1))
Xbar = np.concatenate((one, X), axis = 1)

y = np.array([1.16, 1.3, 1.78, 2, 2.48, 2.57, 2.91])

Xdagger = np.linalg.pinv(Xbar)
w = np.dot(Xdagger, y)
print("E0 = %f, C = %f" %(w[0], w[1]))

newline = np.dot(Xbar, w)
plt.plot(X, y, 'ro')
plt.plot(X, newline)
plt.show()
import numpy as np 
import matplotlib.pyplot as plt 

X = np.array([[-1, 0, 1, 2, 3, 4, 5]]).T
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((X, X*X), axis = 1)
Xbar = np.concatenate((one, Xbar), axis = 1)
Xdagger = np.linalg.pinv(Xbar)

y = np.array([8.5, 2.7, 0.4, -1, -0.2, 3.2, 8.1])

w = np.dot(Xdagger, y)

plt.plot(X, y, 'bo')
#line
X = np.linspace(-1, 5, num = 20).reshape(20, 1)
one = np.ones((20, 1))
Xbar = np.concatenate((X, X*X), axis = 1)
Xbar = np.concatenate((one, Xbar), axis = 1)
newline = np.dot(Xbar, w)

plt.plot(X, newline)
plt.show()
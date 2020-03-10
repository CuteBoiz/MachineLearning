import numpy as np
import matplotlib.pyplot as plt

X = np.array([[147, 150, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis = 1)
y = np.array([49, 50, 51, 54, 58, 59, 60, 62, 63, 64, 66, 67])


w = np.dot(np.linalg.pinv(Xbar), y)
line = np.dot(Xbar, w)

print(line)
plt.plot(X, y, 'ro')
plt.plot(X, line)

plt.axis([140, 190, 45, 75])
plt.xlabel('Height(cm)')
plt.ylabel('Weight(kg)')

plt.show()


import numpy as np 
import matplotlib.pyplot as plt


np.random.seed(22)

means = [[2, 2], [4, 2]]
cov = [[.3, .2], [.2, .3]]
N = 10
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)


plt.plot(X0[:, 0], X0[:, 1], 'bs', markersize = 8)
plt.plot(X1[:, 0], X1[:, 1], 'ro', markersize = 8)
plt.plot([3.5], [2.5],  'k^', markersize = 10, alpha = 0.5)
plt.text(3.6, 2.5, r'?', fontsize=15)
plt.axis('equal')

plt.ylim(0, 3)
plt.xlim(2, 4)
plt.xlabel('$x_1$', fontsize = 15)
plt.ylabel('$x_2$', fontsize = 15)

cur_axes = plt.gca() #get current axes
cur_axes.axes.get_xaxis().set_ticks([])
cur_axes.axes.get_yaxis().set_ticks([])


plt.show()


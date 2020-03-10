import numpy as np 
import matplotlib.pyplot as plt 
from time import time

X = np.random.randn(500, 2)
Xlabel = np.random.randint(0, 2, (500, 1))
z = np.random.randn(2)

red = X[Xlabel.ravel() == 1]
blue = X[Xlabel.ravel() == 0]
plt.scatter(red[:, 0], red[:, 1], 20, 'r')
plt.scatter(blue[:, 0], blue[:, 1], 20, 'b')

print(z)

def dist_pp(z, x):
	d = z - x.reshape(z.shape)
	return np.sum(d*d)

def dist_ps_naive(z, X):
	n = X.shape[0]
	res = np.zeros((1, n))
	for i in range (n):
		res[0][i] = dist_pp(z, X[i])
	return res 

def dist_ps_fast(z, X):
	X2 = np.sum(X*X, 1)
	z2 = np.sum(z*z)
	return X2 + z2 - 2*X.dot(z)

t1 = time()
D1 = dist_ps_naive(z, X)
print('naive: ', time() - t1, 's')

t1 = time()
D2 = dist_ps_fast(z, X)
print('fast: ', time() - t1, 's')

print('Different: ', np.linalg.norm(D1 - D2))

plt.show()
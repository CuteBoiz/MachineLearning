import numpy as np 
import matplotlib.pyplot as plt 
from time import time

#25 random point from 0 - 100
trainData = np.random.randint(0, 100, (25, 2)).astype(np.float32)

#25 random label 
result = np.random.randint(0, 2, (25, 1)).astype(np.float32)


red = trainData[result.ravel() == 1]
blue = trainData[result.ravel() == 0]
newMember = np.random.randint(0, 100, (1, 2)).astype(np.float32)

#compute distance between two vector
def dist_pp(z, x):
	d = z - x.reshape(z.shape)
	return np.sum(d*d)

def dist_ps_naive(z, X):
	res = np.zeros((1, X.shape[0]))
	for i in range (X.shape[0]):
		res[0][i] = dist_pp(z, X[i])
	return res

def dist_ps_fast(z, X):
	X2 = np.sum(X*X, 1)
	z2 = np.sum(z*z)
	return X2 + z2 - 2*X.dot(z)

plt.scatter(red[:, 0], red[:, 1], 100, 'r', 's')
plt.scatter(blue[:, 0], blue[:, 1], 100, 'b', '^')
plt.scatter(newMember[:, 0], newMember[:, 1], 100, 'g', 'o')


t1 = time()
D1 = dist_ps_naive(newMember, trainData)
print('naive point2set, running time: ', time() - t1, 's')

t1 = time()
D2 = dist_ps_fast(newMember, trainData)
print('fast point2set, running time: ', time() - t1, 's')

print('Result differnent: ', np.linalg.norm(D1 - D2))



plt.show()
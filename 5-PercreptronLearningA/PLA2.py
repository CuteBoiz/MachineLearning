import numpy as np 
import matplotlib.pyplot as plt 

np.random.seed(2)

means = [[-1, 0], [1, 0]]
cov = [[.3, .2], [.2, .3]]
N = 10
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)

X = np.concatenate((X0, X1), axis = 0)
y = np.concatenate((np.ones(N), -1*np.ones(N)))

def predict(w, X):
	'''
	w: a 1-d array 
	X: a 2-d array,  each row is a data point
	'''
	return np.sign(X.dot(w))


def PLA(X, y, w_init):
	'''
	X: a 2-d array, each row is a data point
	y: a 1-d array, each row is a label (1/-1) of a X's data point
	w_init: a 1-d array
	'''
	w = w_init
	while True:
		pred = predict(w, X)
		mis_idxs = np.where(np.equal(pred, y) == False)[0] #find indexs of missed points
		num_mis = mis_idxs.shape[0]
		if num_mis == 0:
			return w
		random_id = np.random.choice(mis_idxs, 1)[0]
		w = w + y[random_id]*X[random_id]

def viperceptron(X, y, w_init):
	w = w_init
	w_hist = [w]
	mis_points = []
	while True:
		pred = predict(w, X)
		mis_idxs = np.where(np.equal(pred, y) == False)[0]
		print('miss indexes:', mis_idxs)
		num_mis = mis_idxs.shape[0]
		if num_mis == 0:
			return (w_hist, mis_points)
		random_id = np.random.choice(mis_idxs, 1)[0]
		print('random_id:', random_id)
		break
		mis_points.append(random_id)
		w = w + y[random_id]*X[random_id]
		w_hist.append(w)

np.random.seed(73)
Xbar = np.concatenate((np.ones((2*N, 1)), X), axis = 1)
w_init = np.random.rand(Xbar.shape[1])
print('w init:', w_init)
w, misp = viperceptron(Xbar, y, w_init)
print('Result: ', w[-1])


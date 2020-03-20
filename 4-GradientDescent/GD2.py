import numpy as np 
import matplotlib.pyplot as plt 

def grad(w):
	N = Xbar.shape[0]
	return 1/N * Xbar.T.dot(Xbar.dot(w) - y)

def cost(w):
	N = Xbar.shape[0]
	return .5/N*np.linalg.norm(y - Xbar.dot(w))**2

def GD2(w_init, eta):
	w = [w_init]
	for it in range(100):
		w_new = w[-1] - eta*grad(w[-1])
		if np.linalg.norm(grad(w_new))/len(w_new) < 1e-3:
			break
		w.append(w_new)
	return (w, it)

def draw(X0, Y0, w1, ids, nrows = 2, ncols = 4):
	width = 3*ncols
	height = 3*nrows

	plt.close('all')
	fig,axs=plt.subplots(nrows, ncols, figsize=(width, height))
	for i, k in enumerate(ids):
		r = i//ncols
		c = i%ncols

		b = w1[ids[i]][0]
		w = w1[ids[i]][1]
		x = np.linspace(start = 0, stop = 1, num = 20)
		y = w*x +b
		str0 = 'iter = {}/{}, w = {:.3f}, b = {:.3f}'.format(ids[i], len(w1) - 1, b, w)
		if nrows > 1:
			axs[r, c].plot(X0, Y0, 'bo', markersize = .5)
			axs[r, c].set_xlabel(str0)
			axs[r, c].plot(x, y, 'r')
			axs[r, c].axis([0, 1, 0, 8])
			axs[r, c].plot()
			axs[r, c].tick_params(axis = 'both', which = 'major', labelsize = 13)
		else:
			axs[c].plot(X0, Y0, 'bo', markersize = 5)
			axs[c].set_xlabel(str0)
			axs[c].plot(x, y, 'r')
			axs[c].axis([0, 1, 0, 8])
			axs[c].plot()
			axs[c].tick_params(axis = 'both', which = 'major', labelsize = 8)

	plt.show()




X = np.random.rand(1000)
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X.reshape(-1, 1)), axis = 1)

y = 4 + 3 * X + .5*np.random.rand(1000) #noise added

w_init = np.array([2, 1])
(w1, it1) = GD2(w_init, 1)
ids = [0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 36, 37 ,38, 39]
print('Sol found by GD: w = ', w1[-1], ',\nafter %d iterations.' %(it1+1))
draw(X, y, w1, ids, 4, 4)
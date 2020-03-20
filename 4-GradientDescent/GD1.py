import numpy as np 
import matplotlib.pyplot as plt 

def grad(x):
	return 2*x + 5*np.cos(x)

def cost(x):
	return x**2 + 5*np.sin(x)

def GD1(x0, eta):
	x = [x0]
	for it in range(100):
		x_new = x[-1] - eta*grad(x[-1])
		if abs(grad(x_new)) < 1e-3: #Near 0 number
			break
		x.append(x_new)
	return (x, it)

def draw(x1, ids, nrows = 2, ncols = 4, start = -5.5):
	x0 = np.linspace(start, 5.5, 1000)
	y0 = cost(x0)
	width = 4*ncols
	height = 4*nrows
	
	plt.close('all')
	fig, axs = plt.subplots(nrows, ncols, figsize=(width, height))
	for i,k in enumerate(ids):
		r = i//ncols
		c = i%ncols
		
		x = x1[ids[i]]
		y = cost(x)

		str0 = 'iter {}/{}, grad = {:.3f}'.format(ids[i], len(x1) - 1, grad(x))

		if nrows > 1:
			axs[r, c].plot(x0, y0, 'b')
			axs[r, c].set_xlabel(str0, fontsize = 13)
			axs[r, c].plot(x, y, 'ro', markersize = 7, markeredgecolor = 'k')
			axs[r, c].plot()
			axs[r, c].tick_params(axis='both', which='major', labelsize=13)

		else:
			axs[c].plot(x0, y0, 'b')
			axs[c].set_xlabel(str0, fontsize = 13)
			axs[c].plot(x, y, 'ro', markersize = 7, markeredgecolor = 'k')
			axs[c].plot()
			axs[c].tick_params(axis='both', which='major', labelsize=13)
	plt.show()


(x1, it1) = GD1(5, .1)
ids = [0, 1, 2, 3, 4, 5, 7, 11, 20, 25, 29]
draw(x1, ids, 3)



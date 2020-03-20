import numpy as np 
import matplotlib.pyplot as plt 

x = np.linspace(start = -5.5, stop = 5.5, num =100)
y = x**2 + 10*np.sin(x)

def grad(x):
	return 2*x + 10*np.cos(x)

def cost(x):
	return x**2 + 10*np.sin(x)

def GD3(theta_init, eta, gamma):
	theta = [theta_init]
	v_old = np.zeros_like(theta_init)
	for it in range(100):
		v_new = gamma*v_old + eta*grad(theta[-1])
		theta_new = theta[-1] - v_new
		if np.linalg.norm(grad(theta_new))/np.array(theta_init).size < 1e-3:
			break
		theta.append(theta_new)
		v_old = v_new 
	return theta

def draw(theta, ids, nrows=2, ncols=4):
	x0 = np.linspace(start = -5.5, stop = 5.5, num = 100)
	y0 = cost(x0)

	width = 3.5*ncols
	height = 4*ncols

	plt.close('all')

	fig, axs = plt.subplots(nrows, ncols, figsize = (width, height))

	for i, k in enumerate(ids):
		r = i//ncols
		c = i%ncols

		x = theta[k]
		y = cost(x)
		str0 = 'iter {}/{}, x={:.2f}, y={:.2f}, grad={:.3f}'.format(ids[i], len(theta) - 1, x, y, grad(x))

		if nrows > 1:
			axs[r, c].plot(x0, y0)
			axs[r, c].set_xlabel(str0, fontsize = 8)
			axs[r, c].plot(x, y, 'bo', markersize = 6)
			axs[r, c].tick_params(labelsize=6)
			axs[r, c].set_xticks([-5.5, 0, 5.5])
		else:
			axs[c].plot(x0, y0)
			axs[c].set_xlabel(str0, fontsize = 8)
			axs[c].plot(x, y, 'bo', markersize = 6)
			axs[c].tick_params(labelsize=8)
			axs[r, c].set_xticks([-5.5, 0, 5.5])
	plt.show()


theta = GD3(5, 0.1, 0.9)
ids = [0, 1, 2, 3, 4, 15, 20, 50, 75, 90, 95, 100]
draw(theta, ids, 3, 4)

import numpy as np 
import matplotlib.pyplot as plt 






def grad(x):
	return 2*x + 5*np.cos(x)

def cost(x):
	return x**2 + 5*np.sin(x)

def GD1(x0, eta):
	x = [x0]
	for it in range(100):

		#Draw the table
		x1 = np.linspace(start = -5, stop = 5, num = 30)
		y1 = x1**2 + 5*np.sin(x1) 
		plt.plot(x1, y1)
		plt.axis([-5, 5, -10, 30])
		plt.plot(x[-1], cost(x[-1]), 'bo')
		print("x = %f, y = %f, y' = %f, iter = %d"%(x[-1], cost(x[-1]), grad(x[-1]), it))
		plt.show()
		

		x_new = x[-1] - eta*grad(x[-1])
		if abs(grad(x_new)) < 1e-3: #Near 0 number
			break
		x.append(x_new)
	return (x, it)


#case 1:
(x1, it1) = GD1(-5, .1)
(x2, it2) = GD1(5, .1)

#case 2:
(x3, it3) = GD1(-5, .5)

#case 3:
(x4, it4) = GD1(-5, 0.01)



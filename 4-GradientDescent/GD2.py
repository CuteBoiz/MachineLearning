from sklearn.linear_model import LinearRegression as LR 
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

		plt.plot(X, y, 'bo', markersize = 2)

		temp = w[-1]
		
		x1 = np.linspace(start = 0, stop = 1, num =20)
		#y1 = temp[0][0] + temp[0][1]*x1
		#plt.plot(x1, y1)
		print("b = %f, w = %f, it = %d"%(temp[0][0], temp[1][0], it))

		plt.show()
		w_new = w[-1] - eta*grad(w[-1])
		if np.linalg.norm(grad(w_new))/len(w_new) < 1e-3:
			break
		w.append(w_new)
	return (w, it)



X = np.random.rand(1000)
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X.reshape(-1, 1)), axis = 1)

y = 4 + 3 * X + .5*np.random.rand(1000) #noise

w_init = np.array([[2], [1]])
(w1, it1) = GD2(w_init, 1)




#plt.show()
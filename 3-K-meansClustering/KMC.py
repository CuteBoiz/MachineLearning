from __future__ import print_function
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.backends.backend_pdf import PdfPages
from scipy .spatial.distance import cdist
import random
np.random.seed(18)

means = [[2, 2], [8, 3], [3, 6]]
cov = [[1, 0], [0, 1]]
N = 500
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)

X = np.concatenate((X0, X1, X2), axis = 0)
K = 3
original_label = np.asarray([0]*N + [1]*N + [2]*N).T

plt.scatter(X0[:, 0], X0[:, 1], 20, 'r')
plt.scatter(X1[:, 0], X1[:, 1], 20, 'b')
plt.scatter(X2[:, 0], X2[:, 1], 20, 'g')

def kmeans_init_centroid(X, k):
	return X[np.random.choice(X.shape[0], k, replace = False)]

def kmeans_asign_labels(X, centroids):
	#calculate pairwise distance btw data and centroids
	D = cdist(X, centroids)
	#return index of closest centroid
	return np.argmin(D, axis = 1)

def has_covered(centroids, new_centroids):
	#return True if 2 sets of centroids are the same
	return (set([tuple (a) for a in centroids]) == set([tuple(a) for a in new_centroids]))

def kmeans_update_centroids(X, labels, K):
	centroids = np.zeros((K, X.shape[1]))
	for k in range (K):
		#collect all points that are assigned to the k-th clustter
		Xk = X[labels == k, :]
		centroids[k, :] = np.mean(Xk, axis = 0) #then take average
	return centroids 

def kmeans(X, k):
	centroids = [kmeans_init_centroid(X, k)]
	labels = []
	it = 0
	while True:
		labels.append(kmeans_asign_labels(X, centroids[-1]))
		new_centroids = kmeans_update_centroids(X, labels[-1], K)
		if has_covered(centroids[-1], new_centroids):
			break
		centroids.append(new_centroids)
		it += 1
	return (centroids, labels, it)


def kmeans_display(X, label, filename = 'data.pdf'):
    X0 = X[label == 0, :]
    X1 = X[label == 1, :]
    X2 = X[label == 2, :]
    with PdfPages(filename) as pdf:       
        kwargs = {"markersize": 5, "alpha": .8, "markeredgecolor": 'k'}
        plt.plot(X0[:, 0], X0[:, 1], 'b^', **kwargs)
        plt.plot(X1[:, 0], X1[:, 1], 'go', **kwargs)
        plt.plot(X2[:, 0], X2[:, 1], 'rs', **kwargs)
        plt.axis([-3, 14, -2, 10])
        plt.axis('scaled')
        plt.plot()
        pdf.savefig(bbox_inches='tight')
        plt.show()
    

(centroids, labels, it) = kmeans(X, K)
print('Centers found by out algorithm:\n', centroids[-1])
kmeans_display(X, labels[-1])


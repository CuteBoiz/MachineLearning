import numpy as np 
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

np.random.seed(7)
iris = datasets.load_iris()
iris_X = iris.data
iris_Y = iris.target
print('Label: ', np.unique(iris_Y))

X_train, X_test, Y_train, Y_test = train_test_split(iris_X, iris_Y, test_size = 130)
print('Train size: ', X_train.shape[0], 'Test_size: ', X_test.shape[0])
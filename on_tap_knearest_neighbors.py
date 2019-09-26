# on tap knearest neighbors voi bai toan iris
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

## load iris
iris = datasets.load_iris()
X = iris['data']
y = iris['target']
X_train, X_test, y_train, y_test = train_test_split(X, y)

## knearest neighbors
def dist(X, Z):
    X2 = np.sum(X*X, axis = 1)
    Z2 = np.sum(Z*Z, axis = 1)
    return Z2.reshape(-1, 1) + X2.reshape(1, -1) - 2*(Z.dot(X.T))

# def kneighbors(X_train, X_test, y_train):
di = dist(X_train, X_test)
# print(np.argmin(di, axis = 1).shape)
# print(y_train.shape)
y_pred = y_train[np.argmin(di, axis = 1)]
print(y_pred)
print(y_test)
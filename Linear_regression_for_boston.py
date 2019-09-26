import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

boston = datasets.load_boston()
X = boston['data']
y = boston['target'].reshape(-1, 1)
one = np.ones((X.shape[0], 1))
X = np.concatenate((one, X), axis = 1)

n_train = int(0.8*X.shape[0])
X_train = X[:n_train, :]
X_test = X[n_train:, :]
y_train = y[:n_train]
y_test = y[n_train:]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8)

A = X_train.T.dot(X_train)
B = X_train.T.dot(y_train)
w = np.linalg.pinv(A).dot(B)

y_pred = X_test.dot(w)
print(y_pred[:6].T)

lin = LinearRegression().fit(X_train, y_train)
y_pred1 = lin.predict(X_test)
print(y_pred1[:6].T)

print(y_test[:6].T)

plt.plot(y_test)
plt.plot(y_pred)
plt.show()
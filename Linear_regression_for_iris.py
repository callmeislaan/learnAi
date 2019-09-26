import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
X = iris['data']
y = iris['target']
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis=1)
X_train, X_test, y_train, y_test = train_test_split(Xbar, y)

A = X_train.T.dot(X_train)
B = X_train.T.dot(y_train)
w = np.linalg.pinv(A).dot(B)

w_0 = w[0]
w_1 = w[1]
w_2 = w[2]
w_3 = w[3]
w_4 = w[4]

y_pred = np.round(X_test.dot(w), decimals=0)
print(y_pred)
print(y_test)
print(accuracy_score(y_test, y_pred)*100)
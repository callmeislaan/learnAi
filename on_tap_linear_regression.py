# on tap linear regression voi bai toan du doan gia nha
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

## load boston data
boston = datasets.load_boston()
X = boston['data']
y = boston['target']
one = np.ones((X.shape[0], 1))
X = np.concatenate((one, X), axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y)

## linear regression
w = np.linalg.pinv(X_train.T.dot(X_train)).dot(X_train.T.dot(y_train))
print('w found by self: ', w)
print('y_pred found by self: ')
y_pred = w.dot(X_test.T)
print(y_pred)

## use liblary sklearn
# lireg = LinearRegression().fit(X_train, y_train)
# print('w found by liblary: ', lireg.coef_)
# y_pred = lireg.predict(X_test)
# print('y_pred found by liblary: ')
# print(y_pred)

fig, ax = plt.subplots()
ax.plot(y_test, label = 'y_true')
ax.plot(y_pred, label = 'y_predict')
plt.show()
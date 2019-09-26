import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

boston = datasets.load_boston()
X_train, X_test, y_train, y_test = train_test_split(boston['data'], boston['target'], test_size=0.3)

# use Kneighbors
# knei = KNeighborsClassifier(n_neighbors=10).fit(X_train, y_train)
# y_pred1 = knei.predict(X_test)

# print(y_pred1)
# print('ti le: ', accuracy_score(y_test, y_pred1)*100)

# use linearRegression
# regr = LinearRegression(fit_intercept=False).fit(X_train, y_train)
# y_pred2 = np.round(regr.predict(X_test), decimals=1)
# print(y_test[:10])
# print(y_pred2[:10])

# plt.plot(y_test)
# plt.plot(y_pred2)
# plt.show()

# xu ly boston voi Gradient descent
one = np.ones((X_train.shape[0], 1))
Xbar = np.concatenate((X_train, one), axis = 1)

y_train = y_train.reshape(-1, 1)

def grad(w):
    N = Xbar.shape[0]
    return 1/N*Xbar.T.dot(Xbar.dot(w) - y_train)

def cost(w):
    N = Xbar.shape[0]
    return 0.5/N * np.linalg.norm(y_train - Xbar.dot(w), 2)**2

def GD(w0, grad, eta = 0.1):
    w = [w0]
    for _ in range(100000):
        w_new = w[-1] - eta*grad(w[-1])
        w.append(w_new)
        if np.linalg.norm(grad(w_new), 2) / len(w_new) < 1e-3:
            break
    
    return w[-1]

import check_grad
print(check_grad.check_grad(np.random.rand(Xbar.shape[1], 1), cost, grad))

# w0 = np.random.rand(Xbar.shape[1], 1)
# w = GD(w0, grad, eta=1e-6)
# print(w)
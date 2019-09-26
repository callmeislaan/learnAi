import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

###########################
### load data ###
###########################
iris = datasets.load_iris()
X = iris['data']
y = iris['target']
y = y.reshape(X.shape[0], 1)
one = np.ones((X.shape[0], 1))
X = np.concatenate((X, one), axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
Xbar = X_train.copy()

##############################
### training ###
##############################
def cost(w):
    N = Xbar.shape[0]
    return 0.5/N * np.linalg.norm(y_train - Xbar.dot(w), 2) ** 2

def grad(w):
    N = Xbar.shape[0]
    return 1/N * Xbar.T.dot(Xbar.dot(w) - y_train)

import check_grad
print(check_grad.check_grad(np.random.rand(Xbar.shape[1], 1), cost, grad))

###################################
### Gradient descent batch ###
###################################
def GD(w0, grad, eta = 0.1):
    w = [w0]
    for _ in range(100000):
        w_new = w[-1] - eta*grad(w[-1])
        w.append(w_new)
        if np.linalg.norm(grad(w_new)) / len(w_new) < 1e-3:
            break
    return w


###################################
### Gradent descent with momentum
def GD_momentum(w0, grad, eta = 1e-4, gamma = 0.9):
    w = [w0]
    v_old = np.zeros_like(w0)
    for i in range(10000):
        v_new = eta*grad(w[-1]) + gamma*v_old
        w_new = w[-1] - v_new
        w.append(w_new)
        if np.linalg.norm(w_new)/len(w_new) < 1e-4:
            break
        v_old = v_new
    return w, i


####################################
### Gradient descent with NAG
def GD_NAG(w0, grad, eta = 1e-4, gamma = 0.9):
    w = [w0]
    v_old = np.zeros_like(w0)
    for i in range(10000):
        v_new = eta*grad(w[-1] - gamma*v_old) + gamma*v_old
        w_new = w[-1] - v_new
        w.append(w_new)
        v_old = v_new
        if np.linalg.norm(grad(w_new))/len(w_new) < 1e-4:
            break
    return w, i

################################
### show result ###
################################

####################################
### Gradient descent
print('Gradient descent: ')
# w0 = np.asarray([[10], [10], [10], [10], [10]])
w0 = np.random.rand(Xbar.shape[1], 1)
# w = GD(w0, grad, 1e-5)
w1, i1 = GD_momentum(w0, grad, eta=1e-2)
print('momentum')
print(w1[-1], i1)
w2, i2 = GD_NAG(w0, grad, eta=1e-2)
print('NAG')
print(w2[-1], i2)
y_pred = []
for x in X_test:
    y_p = np.round(np.sum(wi*xi for wi, xi in zip(w2[-1], x)), decimals=0)
    y_pred.append(y_p)
# print(y_pred)
print('right rate by GD_: ', accuracy_score(y_test, y_pred)*100)

#######################################
### Linear Regression
print('\nLinear Regression: ')
regr = LinearRegression(fit_intercept=False).fit(X_train, y_train)
w = regr.coef_
y_pred = np.round(regr.predict(X_test),decimals=0)
print('w found by regr: ', w)
print('right rate by Linear: ', accuracy_score(y_test, y_pred)*100)
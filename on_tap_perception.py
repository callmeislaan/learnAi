## on tap perception voi bai toan ung thu vu
import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron

## load data
breast_cancer = datasets.load_breast_cancer()
X = breast_cancer['data']
y = breast_cancer['target']
y = np.where(y == 0, -1, y)
X_train, X_test, y_train, y_test = train_test_split(X, y)

## perception

def h(w, x):
    return np.sign(w.dot(x.T))

def has_converged(w, X, y):
    return np.array_equal(h(w, X), y)

def perception(w_init, X, y, max_count = 10000):
    w = [w_init]
    count = 0
    N = X.shape[0]
    while count < max_count:
        mix_id = np.random.permutation(N)
        for i in mix_id:
            xi = X[i, :].reshape(1, -1)
            yi = y[i]
            count += 1
            if h(w[-1], xi) != yi:
                w_new = w[-1] + yi*xi
                w.append(w_new)
        if has_converged(w[-1], X, y):
            break
    return w

w_init = np.random.randn(1, X.shape[1])
w = perception(w_init, X_train, y_train)

def predict(w, X):
    return np.sign(w.dot(X.T))

y_pred = predict(w[-1], X_test)
print(accuracy_score(y_test, y_pred.T)*100)

per = Perceptron().fit(X_train, y_train)
y_pred = per.predict(X_test)
print(accuracy_score(y_test, y_pred)*100)
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

### load data ###
iris = datasets.load_iris()
X = iris['data']
y = iris['target']
one = np.ones((X.shape[0], 1))
X = np.concatenate((one, X), axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

### training ###

def h(w, x):
    return np.sign(w.T.dot(w))

def has_converged(X, y, w):
    return np.array_equal(h(w, X), y)

def myPLA(X, y, w0):
    w = [w0]
    N = X_train.shape[0]
    while True:
        rd_id = np.random.permutation(N)
        for i in range(N):
            true_id = rd_id[i]
            xi = X_train[true_id, :]
            yi = y_train[true_id]
            if h(w[-1], xi)[0] != yi:
                w_new = w[-1] + yi*xi
                w.append(w_new)
        
        if has_converged(X, y, w[-1]):
            break
    return w
w0 = np.random.randn(X_train.shape[1], 1)
w = myPLA(X_train, y_train, w0)
print(w[-1])
                
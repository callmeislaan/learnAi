## on tap softmax voi bai toan breast cancer
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from scipy.sparse import coo_matrix

## load data
dt = datasets.load_breast_cancer()
X = dt['data']
y = dt['target']

def convert_labels(y, C):
    Y = coo_matrix((np.ones_like(y), (y, np.arange(len(y)))), shape = (C, len(y))).toarray()
    return Y

X_train, X_test, y_train, y_test = train_test_split(X, y)
X_train = X_train.T
X_test = X_test.T
Y_train = convert_labels(y_train, C = 2)
Y_test = convert_labels(y_test, C = 2)

## training
def softmax(Z):
    e_Z = np.exp(Z)
    return e_Z / np.sum(e_Z, axis = 0)

def softmax_stable(Z):
    e_Z = np.exp(Z - np.max(Z, axis = 0, keepdims = True))
    return e_Z / np.sum(e_Z, axis = 0)

def softmax_regression(W_init, X, Y, eta = 1e-3, max_count = 10000):
    count = 0
    W = [W_init]
    check_w_after = 20
    N = X.shape[1]
    d = X.shape[0]
    C = Y.shape[0]
    while count < max_count:
        mix_id = np.random.permutation(N)
        for i in mix_id:
            xi = X[:, i].reshape(d, 1)
            yi = Y[:, i].reshape(C, 1)
            zi = W[-1].T.dot(xi)
            ai = softmax_stable(zi) # dung softmax se bi tran so
            ei = yi - ai
            W_new = W[-1] + eta * xi.dot(ei.T)
            count += 1
            if count%check_w_after == 0:
                if np.linalg.norm(W_new - W[-check_w_after]) < 1e-5:
                    return W, count
            W.append(W_new)
    
    return W, count

C = 2
d = X_train.shape[0]
W_init = np.random.randn(d, C)
W, count = softmax_regression(W_init, X_train, Y_train)
print(W[-1])
print(count)

Z = W[-1].T.dot(X_test)
y_pred = np.argmax(softmax_stable(Z), axis = 0)
print(accuracy_score(y_test, y_pred)*100)
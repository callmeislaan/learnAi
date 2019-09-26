import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

### load data ###

iris = datasets.load_iris()
X = iris['data']
y = iris['target']
one = np.ones((X.shape[0], 1))
X = np.concatenate((X, one), axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

y_train = y_train.reshape(-1, 1)

X_y_train = np.concatenate((X_train, y_train), axis = 1)

### training ###

def grad(X_y_train_shuffle, w, i, mini_size = 25):
    Xmn = X_y_train_shuffle[i:i+mini_size, :-1]
    ymn = X_y_train_shuffle[i:i+mini_size, -1].reshape(-1, 1)
    N = mini_size
    return 1/N*Xmn.T.dot(Xmn.dot(w) - ymn)

def MGD(w0, grad, eta = 1e-4):
    w = [w0]
    N = X_train.shape[0]
    mini_size = 5
    iter_w = 5
    w_last_check = w0
    count = 0
    for it in range(1000):
        np.random.shuffle(X_y_train)
        i = 0
        while i < N:
            count += 1
            v_new = eta*grad(X_y_train, w[-1], i, mini_size)
            w_new = w[-1] - v_new
            w.append(w_new)
            if count % iter_w == 0:
                w_this_check = w_new
                if np.linalg.norm(w_this_check - w_last_check)/len(w_new) < 1e-4:
                    return (w, it)
                w_last_check = w_this_check
            i += mini_size
    return (w, it)

## MGD momentum

def MGD_momentum(w0, grad, eta = 1e-4, gamma = 0.9):
    w = [w0]
    N = X_train.shape[0]
    mini_size = 5
    v_old = np.zeros_like(w0)
    iter_w = 5
    w_last_check = w0
    count = 0
    for it in range(1000):
        np.random.shuffle(X_y_train)
        i = 0
        while i < N:
            count += 1
            v_new = eta*grad(X_y_train, w[-1], i, mini_size) + gamma*v_old
            w_new = w[-1] - v_new
            v_old = v_new
            w.append(w_new)
            if count % iter_w == 0:
                w_this_check = w_new
                if np.linalg.norm(w_this_check - w_last_check)/len(w_new) < 1e-4:
                    return (w, it)
                w_last_check = w_this_check
            i += mini_size
    return (w, it)

## MGD NAG
def MGD_NAG(w0, grad, eta = 1e-4, gamma = 0.9):
    w = [w0]
    N = X_train.shape[0]
    mini_size = 5
    v_old = np.zeros_like(w0)
    iter_w = 5
    w_last_check = w0
    count = 0
    for it in range(1000):
        np.random.shuffle(X_y_train)
        i = 0
        while i < N:
            count += 1
            v_new = eta*grad(X_y_train, w[-1] - gamma*v_old, i, mini_size) + gamma*v_old
            w_new = w[-1] - v_new
            v_old = v_new
            w.append(w_new)
            if count % iter_w == 0:
                w_this_check = w_new
                if np.linalg.norm(w_this_check - w_last_check)/len(w_new) < 1e-4:
                    return (w, it)
                w_last_check = w_this_check
            i += mini_size
    return (w, it)





### show result ###
def find_y_pred(w):
    y_pred = []
    for x in X_test:
        y_p = np.round(sum(wi*xi for wi, xi in zip(w[-1], x)), decimals = 0)
        y_pred.append(y_p)
    return y_pred


w0 = np.random.rand(X_train.shape[1], 1)
w, it = MGD(w0, grad)
print(w[-1], it, len(w))
print('right rate found by MGD: ', accuracy_score(y_test, find_y_pred(w))*100)

## MGD momentum
w1, it1 = MGD_momentum(w0, grad)
print(w1[-1], it1, len(w1))
print('right rate found by MGD_momentum: ', accuracy_score(y_test, find_y_pred(w1))*100)


## MGD NAG
w2, it2 = MGD_NAG(w0, grad)
print(w2[-1], it2, len(w2))
print('right rate found by MGD_NAG: ', accuracy_score(y_test, find_y_pred(w2))*100)
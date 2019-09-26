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

### training ###

# single gradient
def sgrad(w, i, rd_id):
    true_i = rd_id[i]
    xi = X_train[true_i, :]
    yi = y_train[true_i]
    a = xi.dot(w) - yi
    return (xi*a).reshape(-1, 1)

## SGD ##
def SGD(w0, sgrad, eta = 1e-3):
    w = [w0]
    w_last_check = w0
    N = X_train.shape[0]
    iter_w = 10
    count = 0
    for it in range(10):
        rd_id = np.random.permutation(N)
        for i in range(N):
            count += 1
            w_new = w[-1] - eta*sgrad(w[-1], i, rd_id)
            w.append(w_new)
            if count%iter_w == 0:
                w_this_check = w_new
                if np.linalg.norm(w_this_check - w_last_check)/len(w_new) < 1e-4:
                    return (w, it)
                w_last_check = w_this_check
    return (w, it)


## SGD momentum
def SGD_momentum(w0, sgrad, eta = 1e-4, gamma = 0.9):
    w = [w0]
    w_last_check = w0
    iter_w = 10
    N = X_train.shape[0]
    count = 0
    v_old = np.zeros_like(w0)
    for i in range(10):
        rd_id = np.random.permutation(N)
        for i in range(N):
            count += 1
            v_new = eta*sgrad(w[-1], i, rd_id) + gamma*v_old
            w_new = w[-1] - v_new
            w.append(w_new)
            v_old = v_new
            if count%iter_w == 0:
                w_this_check = w_new
                if np.linalg.norm(w_this_check - w_last_check)/len(w_new) < 1e-4:
                    return (w, it)
                w_last_check = w_this_check

    return (w, it)

## SGD NAG
def SGD_NAG(w0, sgrad, eta=1e-4, gamma=0.9):
    w = [w0]
    w_last_check = w0
    N = X_train.shape[0]
    iter_w = 10
    v_old = np.zeros_like(w0)
    count = 0
    for it in range(10):
        rd_id = np.random.permutation(N)
        for i in range(N):
            v_new = eta*sgrad(w[-1] - gamma*v_old, i, rd_id) + gamma*v_old
            w_new = w[-1] - v_new
            w.append(w_new)
            v_old = v_new
            count += 1
            if count % iter_w == 0:
                w_this_check = w_new
                if np.linalg.norm(w_this_check - w_last_check)/len(w_new) < 1e-4:
                    return (w, it)
                w_last_check = w_this_check
            
    return (w, it)

### show result ###
def find_y_pred(w):
    y_pred = []
    for i in range(X_test.shape[0]):
        y_p = sum(wi*xi for wi, xi in zip(w[-1], X_test[i, :]))
        y_pred.append(np.round(y_p, decimals=0))
    return y_pred
## SGD
w0 = np.random.rand(X_train.shape[1], 1)
w, it = SGD(w0, sgrad)
print(w[-1], it, len(w))
print('right rate by SGD: ', accuracy_score(y_test, find_y_pred(w))*100)

## SGD momentum
w1, it1 = SGD_momentum(w0, sgrad)
print(w1[-1], it1, len(w1))
y_pred1 = find_y_pred(w1)
print('right rate by SGD_momentum: ', accuracy_score(y_test, y_pred1)*100)

## SGD NAG
w2, it2 = SGD_NAG(w0, sgrad)
print(w2[-1], it2, len(w2))
y_pred2 = find_y_pred(w2)
print('right rate by SGD_NAG: ', accuracy_score(y_test, y_pred2)*100)
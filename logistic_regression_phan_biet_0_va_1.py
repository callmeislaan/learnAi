import numpy as np
import matplotlib.pyplot as plt
import gzip
import os
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from time import time

# load data
def load_mnist(path, kind = 'train'):
    train_labels = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
    train_images = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)

    with gzip.open(train_labels, mode='rb') as lbl:
        labels = np.frombuffer(lbl.read(), dtype=np.uint8, offset=8)
    with gzip.open(train_images, mode='rb') as img:
        images = np.frombuffer(img.read(), dtype = np.uint8, offset = 16).reshape(-1, 28*28)

    return labels, images

labels, images = load_mnist('Data/mnist/')
X0 = images[labels == 0, :]
X1 = images[labels == 1, :]
y0 = labels[labels == 0]
y1 = labels[labels == 1]
X = np.concatenate((X0, X1), axis = 0)
y = np.concatenate((y0, y1), axis = 0)

def sigmoid(s):
    return 1 / (1 + np.exp(-s))

def logistic_sigmoid_regression(w0, X, y, eta = 1e-3):
    w = [w0]
    last_check_w = w0
    iter_w = 10
    N = X.shape[0]
    d = X.shape[1]
    count = 0
    check_w_after = 20
    for it in range(iter_w):
        mix_id = np.random.permutation(N)
        for i in mix_id:
            xi = X[i, :].reshape(d, 1)
            yi = y[i]
            zi = sigmoid(w[-1].T.dot(xi))
            w_new = w[-1] + eta*(yi - zi)*xi
            w.append(w_new)
            count += 1
            if count%check_w_after == 0:
                this_check_w = w_new
                if np.linalg.norm(this_check_w - last_check_w) < 1e-3:
                    return w, it
                last_check_w = this_check_w
            
    return w, it


y_test, X_test = load_mnist('Data/mnist/', kind = 't10k')
X_test_0 = X_test[y_test == 0, :]
y_test_0 = y_test[y_test == 0]
X_test_1 = X_test[y_test == 1, :]
y_test_1 = y_test[y_test == 1]
X_test = np.concatenate((X_test_0, X_test_1), axis = 0)
y_test = np.concatenate((y_test_0, y_test_1), axis = 0)
print(X_test.shape)
print(y_test.shape)

t1 = time()
w0 = np.random.randn(X.shape[1], 1)
w, it = logistic_sigmoid_regression(w0, X, y)
# print(w[-1])
# print(it)
y_pred = np.round(sigmoid(w[-1].T.dot(X_test.T)), decimals=0)
print(accuracy_score(y_test, y_pred.T)*100, 'time: ', time() - t1)

t1 = time()
logistic = LogisticRegression(C=1e5).fit(X, y)
y_pred = logistic.predict(X_test)
print(accuracy_score(y_test, y_pred)*100, 'time: ', time() - t1)
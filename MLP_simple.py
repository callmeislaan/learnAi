import numpy as np
import matplotlib.pyplot as plt
import math

## create data
N = 100 # number of point per class
d0 = 2  # dimensionality
C = 3   # number of classes
def create_data(N, d0, C):
    X = np.zeros((d0, N*C)) # data matrix
    y = np.zeros(N*C, dtype='uint8')    # class label
    for j in range(C):
        ix = range(N*j, N*(j+1))
        r = np.linspace(0.0, 1, N)  # radius
        t = np.linspace(j*4, (j+1)*4, N) + np.random.randn(N)*0.2   # theta
        X[:, ix] = np.c_[r*np.sin(t), r*np.cos(t)].T
        y[ix] = j
    return X, y

## display data
def display_data(X, y):
    plt.plot(X[0, :N], X[1, :N], 'bs', 'markersize = 7')
    plt.plot(X[0, N:2*N], X[1, N:2*N], 'ro', 'markersize = 7')
    plt.plot(X[0, 2*N:], X[1, 2*N:], 'g^', 'markersize = 7')
    plt.xlim([-1.5, 1.5])
    plt.ylim([-1.5, 1.5])
    cur_axes = plt.gca()
    cur_axes.axes.get_xaxis().set_ticks([])
    cur_axes.axes.get_yaxis().set_ticks([])
    plt.show()

X, y = create_data(N, d0, C)
# display_data(X, y)

def softmax(V):
    e_V = np.exp(V - np.max(V, axis = 0, keepdims = True))
    Z = e_V / e_V.sum(axis = 0)
    return Z

## one-hot coding
from scipy import sparse
def convert_labels(y, C = 3):
    Y = sparse.coo_matrix((np.ones_like(y), (y, np.arange(len(y)))), shape = (C, len(y))).toarray()
    return Y

## cost or lass function
def cost(Y, Yhat):
    return -np.sum(Y*np.log(Yhat))/Y.shape[1]

def MLP():
    d1 = 100 # = h    # size of hidden layer
    d2 = 3 # = c
    W1 = 0.01*np.random.randn(d0, d1)
    b1 = np.zeros((d1, 1))
    W2 = 0.01*np.random.randn(d1, d2)
    b2 = np.zeros((d2, 1))
    Y = convert_labels(y)
    N = X.shape[1]
    eta = 1 # learning rate
    for i in range(10000):
        ## feedforward
        Z1 = W1.T.dot(X) + b1
        A1 = np.maximum(Z1, 0)
        Z2 = W2.T.dot(A1) + b2
        Yhat = softmax(Z2)

        # print loss after each 1000 iterations
        if i % 1000 == 0:
            loss = cost(Y, Yhat)
            print('iter %d, loss: %f'%(i, loss))
        
        ## backpropagation
        E2 = (Yhat - Y) / N
        dW2 = np.dot(A1, E2.T)
        db2 = np.sum(E2, axis = 1, keepdims=True)
        E1 = np.dot(W2, E2)
        E1[Z1 <= 0] = 0 # gradient of ReLU
        dW1 = np.dot(X, E1.T)
        db1 = np.sum(E1, axis = 1, keepdims = True)

        ## gradient descent update
        W1 += -eta*dW1
        b1 += -eta*db1
        W2 += -eta*dW2
        b2 += -eta*db2
MLP()
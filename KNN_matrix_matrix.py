import numpy as np
from time import time

N = 1000
d = 1000
M = 100
X = np.random.randn(N, d)
Z = np.random.randn(M, d)

def dist_pp(z, x):
    d = z - x
    return np.sum(d*d)

def dist_ps_fast(z, X):
    z2 = np.sum(z*z)
    X2 = np.sum(X*X, axis = 1)
    return z2 + X2 - 2*X.dot(z)

def dist_naive(Z, X):
    res = np.zeros((Z.shape[0], X.shape[0]))
    for i in range(M):
        res[i] = dist_ps_fast(Z[i], X)
    return res
def dist_naive_1(Z, X):
    res = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            res[i, j] = dist_pp(Z[i], X[j])
    return res

def dist_fast(Z, X):
    Z2 = np.sum(Z*Z, axis = 1)
    X2 = np.sum(X*X, axis = 1)
    return Z2.reshape(-1, 1) + X2.reshape(1, -1) - 2*Z.dot(X.T)

t1 = time()
D1 = dist_naive(Z, X)
print(time() - t1)

t2 = time()
D2 = dist_fast(Z, X)
print(time() - t2)

t3 = time()
D3 = dist_naive_1(Z, X)
print(time() - t3)

print(np.linalg.norm(D1 - D2))
print(np.linalg.norm(D3 - D1))
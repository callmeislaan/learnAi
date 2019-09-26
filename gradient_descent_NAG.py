import numpy as np

np.random.seed(2)

X = np.random.rand(1000, 1)
y = 4 + 3*X + 0.2*np.random.randn(1000, 1)
print(y.shape)

one = np.ones((X.shape[0], 1))

Xbar = np.concatenate((one, X), axis = 1)

def cost(w):
    N = Xbar.shape[0]
    return 0.5/N*np.linalg.norm(y - Xbar.dot(w), 2)**2

def grad(w):
    N = Xbar.shape[0]
    return 1/N*Xbar.T.dot(Xbar.dot(w) - y)

import check_grad
print(check_grad.check_grad(np.random.rand(Xbar.shape[1], 1), cost, grad))

def GD_NAG(w_init, grad, eta, gamma):
    w = [w_init]
    v = [np.zeros_like(w_init)]
    for _ in range(100):
        v_new = gamma*v[-1] + eta*grad(w[-1] - gamma*v[-1])
        w_new = w[-1] - v_new
        if np.linalg.norm(grad(w_new)) / len(w_new) < 1e-3:
            break
        w.append(w_new)
        v.append(v_new)
    
    return w[-1]

w = GD_NAG(np.random.rand(2, 1), grad, 0.1, 0.9)
print(w)

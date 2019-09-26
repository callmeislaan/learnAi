import numpy as np

X = np.random.rand(1000, 1)
y = 4 + 3*X + 0.2*np.random.randn(1000, 1)

one = np.ones((X.shape[0], 1))
X = np.concatenate((one, X), axis = 1)

def grad(w):
    N = X.shape[0]
    return 1/N * X.T.dot(X.dot(w) - y)

def myGD(w0, grad, eta = 1e-3):
    w = [w0]
    for i in range(100000):
        w_new = w[-1] - eta * grad(w[-1])
        w.append(w_new)
        if np.linalg.norm(grad(w_new))/len(w_new) < 1e-3:
            break
    return w, i

w0 = np.random.rand(X.shape[1], 1)
w, i = myGD(w0, grad)
print(w[-1])
print(i)

## momentum
def myMGD(w0, grad, eta = 1e-3, gamma = 0.9):
    v_old = np.zeros_like(w0)
    w = [w0]
    for i in range(100000):
        v_new = eta*grad(w[-1]) + gamma*v_old
        w_new = w[-1] - v_new
        if np.linalg.norm(grad(w_new)) / len(w_new) < 1e-3:
            break
        w.append(w_new)
        v_old = v_new
    return w, i

w0 = np.random.rand(X.shape[1], 1)
w, i = myMGD(w0, grad)
print(w[-1])
print(i)

## NAG
def myNAG(w0, grad, eta = 1e-3, gamma = 0.9):
    v_old = np.zeros_like(w0)
    w = [w0]
    for i in range(100000):
        v_new = eta*grad(w[-1] - gamma*v_old) + gamma*v_old
        w_new = w[-1] - v_new
        if np.linalg.norm(grad(w_new))/len(w_new) < 1e-3:
            break
        w.append(w_new)
        v_old = v_new
    return w, i

w0 = np.random.rand(X.shape[1], 1)
w, i = myMGD(w0, grad)
print(w[-1])
print(i)
        
import numpy as np

X = np.random.rand(1000, 1)
y = 4 + 3*X + 0.2*np.random.randn(1000, 1)
one = np.ones((X.shape[0], 1))
X = np.concatenate((one, X), axis = 1)

def grad(w, true_id, i, m):
    xim = X[true_id[i:m], :]
    yim = y[true_id[i:m], :]
    return 1/m * xim.T.dot(xim.dot(w) - yim)

def myMGD(w0, grad, m = 50, eta = 0.1):
    w = [w0]
    N = X.shape[0]
    it_check_w = 10
    last_check_w = w0
    count = 0
    for it in range(1000):
        true_id = np.random.permutation(N)
        i = 0
        while i < N:
            w_new = w[-1] - eta * grad(w[-1], true_id, i, m)
            w.append(w_new)
            count += 1
            if count%it_check_w == 0:
                this_check_w = w_new
                if np.linalg.norm(this_check_w - last_check_w) / len(this_check_w) < 1e-5:
                    return w, it
                last_check_w = this_check_w
            i += m
    return w, it

w0 = np.random.rand(X.shape[1], 1)
w, it = myMGD(w0, grad)
print(w[-1])
print(it)
import numpy as np

X = np.random.rand(1000, 1)
y = 4 + 3*X + 0.2*np.random.randn(1000, 1)

one = np.ones((X.shape[0], 1))
X = np.concatenate((one, X), axis = 1)

# single gradient
def sgrad(w, true_id, i):
    xi = X[true_id[i], :].reshape(1, -1)
    yi = y[true_id[i], :].reshape(1, -1)
    return xi.T.dot(xi.dot(w) - yi)

def mySGD(w0, sgrad, eta = 1e-2):
    w = [w0]
    N = X.shape[0]
    w_last_check = w0
    it = 10
    count = 0
    for i in range(10):
        true_id = np.random.permutation(N)
        for j in range(N):
            count += 1
            w_new = w[-1] - eta*sgrad(w[-1], true_id, j)
            w.append(w_new)
            if count%it == 0:
                w_this_check = w_new
                if np.linalg.norm(w_this_check - w_last_check) / len(w_this_check) < 1e-5:
                    return w, i
                w_last_check = w_this_check
    return w, i

w0 = np.random.rand(X.shape[1], 1)
# w0 = np.array([[2], [1]])
w, i = mySGD(w0, sgrad)
print(w[-1])
print(i)
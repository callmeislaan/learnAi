import numpy as np
import matplotlib.pyplot as plt

# np.random.seed(2)
# create data
mean = np.array([[2, 2], [5, 5]])
cov = np.array([[.3, .2], [.2, .3]])
N = 10
X0 = np.random.multivariate_normal(mean[0], cov, N)
X1 = np.random.multivariate_normal(mean[1], cov, N)
X = np.concatenate((X0, X1), axis = 0)
one = np.ones((X.shape[0], 1))
X = np.concatenate((one, X), axis = 1)
y = np.array([1]*N + [-1]*N).reshape(1, -1)
def h(w, xi):
    return np.sign(w.dot(xi.T))

def has_converged(w, X, y):
    return np.array_equal(h(w, X), y)

def myPLA(w0, X, y):
    w = [w0]
    N = X.shape[0]
    while True:
    # for _ in range(1000):
        mix_id = np.random.permutation(N)
        for i in range(N):
            xi = X[mix_id[i], :]
            yi = y[:, mix_id[i]]
            if h(w[-1], xi) != yi:
                w_new = w[-1] + yi*xi
                w.append(w_new)
        if has_converged(w[-1], X, y):
            break
    return w

w0 = np.random.rand(1, X.shape[1])
w = myPLA(w0, X, y)
print(w[-1])
print(h(w[-1], X).astype(int))
print(y)

a = np.arange(0, 10, 0.1)
w_0 = w[-1][:, 0]
w_1 = w[-1][:, 1]
w_2 = w[-1][:, 2]
ax = -(a*w_1 + w_0)/w_2
plt.plot(a, ax)
plt.plot(X0[:, 0], X0[:, 1], 'ro')
plt.plot(X1[:, 0], X0[:, 1], 'b^')

plt.show()

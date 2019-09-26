import numpy as np
import matplotlib.pyplot as plt
np.random.seed(2)

# create data

means = [[2, 2], [4, 2]]
cov = [[.7, 0], [0, .7]]
N = 20
X0 = np.random.multivariate_normal(means[0], cov, N).T
X1 = np.random.multivariate_normal(means[1], cov, N).T
X = np.concatenate((X0, X1), axis = 1)

# plt.plot(X0[0, :], X0[1, :], 'rs')
# plt.plot(X1[0, :], X1[1, :], 'b^')
# plt.show()

one = np.ones((1, X.shape[1]))
X = np.concatenate((one, X), axis = 0)
y = np.array([0]*N + [1]*N)

def sigmoid(s):
    return 1/(1 + np.exp(-s))

def logistic_sigmoid_regression(w0, X, y, eta = 0.05, max_count = 100000):
    w = [w0]
    count = 0
    N = X.shape[1]
    d = X.shape[0]
    check_w_after = 20
    while count < max_count:
        mix_id = np.random.permutation(N)
        for i in mix_id:
            count += 1
            xi = X[:, i].reshape(d, 1)
            yi = y[i]
            zi = sigmoid(w[-1].T.dot(xi))
            w_new = w[-1] + eta*(yi - zi)*xi
            w.append(w_new)
            if count%check_w_after == 0:
                if np.linalg.norm(w_new - w[-check_w_after]) < 1e-3:
                    return w, count
    
    return w, count

d = X.shape[0]
w0 = np.random.randn(d, 1)
w, count = logistic_sigmoid_regression(w0, X, y)
print(w[-1])
print(count)

print(sigmoid(w[-1].T.dot(X))*100)

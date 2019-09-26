import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2)

# X = np.array([[0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 
#               2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50]])
# y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])

# # extended data 
# X = np.concatenate((np.ones((1, X.shape[1])), X), axis = 0)

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

def logistic_sigmoid_regression(w0, X, y, eta = 1e-3, max_count = 10000):
    w = [w0]
    N = X.shape[1]
    check_w_after = 20
    count = 0
    d = X.shape[0]
    while count < max_count:
        mix_i = np.random.permutation(N)
        for i in mix_i:
            count += 1
            xi = X[:, i].reshape(d, 1)
            yi = y[i]
            zi = sigmoid(w[-1].T.dot(xi))
            w_new = w[-1] + eta*(yi - zi)*xi
            w.append(w_new)
            if count%check_w_after == 0:
                if np.linalg.norm(w_new - w[-check_w_after]) / len(w_new) < 1e-4:
                    return w, count

    return w, count

w0 = np.random.randn(X.shape[0], 1)
w, c = logistic_sigmoid_regression(w0, X, y, 0.05)
print(w[-1])
print(c)
print(sigmoid(w[-1].T.dot(X)))

# X0 = X[1, np.where(y == 0)][0]
# y0 = y[y==0]
# X1 = X[1, np.where(y == 1)][0]
# y1 = y[y == 1]
# plt.plot(X0, y0, 'ro', markersize = 8)
# plt.plot(X1, y1, 'bs', markersize = 8)

# xx = np.linspace(0, 6, 1000)
# w0 = w[-1][0][0]
# w1 = w[-1][1][0]
# threshold = -w0/w1
# yy = sigmoid(w0 + w1*xx)
# plt.axis([-2, 8, -1, 2])
# plt.plot(xx, yy, 'g-', linewidth = 2)
# plt.plot(threshold, 0.5, 'y^', markersize = 8)
# plt.xlabel('studying hours')
# plt.ylabel('predicted probatility of pass')
# plt.show()
# dung thuat toan SGD de giai bai toan LinearRegression
import numpy as np

X = np.random.rand(1000, 1)
y = 4 + 3*X + 0.2*np.random.randn(1000, 1)

one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis = 1)

# single point gradient
def sgrad(w, i, rd_id):
    true_i = rd_id[i]
    xi = np.array([Xbar[true_i, :]])
    yi = y[true_i]
    a = xi.dot(w) - yi
    # print(xi.T*a)
    # return (xi*a).reshape(2, 1)
    return xi.T*a

def SGD(w_init, sgrad, eta = 0.1):
    w = [w_init]
    w_last_check = w_init
    iter_check_w = 10
    N = X.shape[0]
    count = 0
    for _ in range(10):
        #shuffle data
        rd_id = np.random.permutation(N)
        for i in range(N):
            count += 1
            g = sgrad(w[-1], i, rd_id)
            w_new = w[-1] - eta*g
            w.append(w_new)
            if count%iter_check_w == 0:
                w_this_check = w_new
                if np.linalg.norm(w_this_check - w_last_check) / len(w_init) < 1e-3:
                    return w
                w_last_check = w_this_check
        
    return w

w = SGD(np.random.rand(2, 1), sgrad)
print(w[-1])
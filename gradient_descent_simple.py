# Xét hàm số f(x)=x^2+5sin(x) với đạo hàm f′(x)=2x+5cos(x)
import math
import numpy as np
import matplotlib.pyplot as plt
import check_grad

# # tinh dao ham 
# def grad(x):
#     return 2*x + 5*np.cos(x)

# # tinh gia tri ham so
# def cost(x):
#     return x**2 + 5*np.sin(x)

# # ham thuc hien thuat toan
# def myGD1(eta, x0):
#     x = [x0]
#     for i in range(100):
#         x_new = x[-1] - eta*grad(x[-1])
#         if abs(grad(x_new)) < 1e-3:
#             break
#         x.append(x_new)
#     return x[-1], i

# x, i = myGD1(0.1, 5)
# print('cost = %f, grad = %f, i = %d' %(grad(x), cost(x), i))

# doi voi ham nhieu bien: thuc hien voi bai toan trong bai linearRegression
np.random.seed(2)

X = np.random.rand(1000, 1)
y = 4 + 3*X + 0.2*np.random.randn(1000, 1)

# buiding Xbar
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis = 1)

A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
w_lr = np.dot(np.linalg.pinv(A), b)
print('solution found by formula: w = ', w_lr.T)

# display result
w = w_lr
w0 = w[0][0]
w1 = w[1][0]
x0 = np.linspace(0, 1, 2, endpoint=True)
y0 = w0 + w1*x0

# draw the fitting line
# plt.plot(X.T, y.T, 'b.')    #data
# plt.plot(x0, y0, 'y', linewidth = 2)    # the fiiting line
# plt.axis([0, 1, 0, 10])
# plt.show()

# dao ham
def grad(w):
    N = Xbar.shape[0]
    return 1/N * Xbar.T.dot(Xbar.dot(w) - y)

def cost(w):
    N = Xbar.shape[0]
    return 0.5/N * np.linspace.norm(y - Xbar.dot(w), 2)**2

# print(check_grad.check_grad(w, cost, grad))
def myGD(w_init, grad, eta):
    w = [w_init]
    for i in range(100):
        w_new = w[-1] - eta*grad(w[-1])
        if np.linalg.norm(grad(w_new)) / len(w_new) < 1e-3:
            break
        w.append(w_new)
    return w, i

w_init = np.array([[2], [1]])
w1, it1 = myGD(w_init, grad, 1)
print('Solution found by GD: w = ', w1[-1].T, ',\nafter %d iterations.' %(it1+1))
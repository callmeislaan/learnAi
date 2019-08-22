### bai 1
# f(x, y) = x^2 + y^2
# f'(x, y) = [2x, 2y]


### bai 2
# f(x, y) = (x^2 + y - 7)^2 + (x - y + 1)^2
# f'(x, y) = [2*(x^2 + y - 7)*2*x + 2*(x - y + 1), 2*(x^2 + y - 7) + 2*(x - y + 1)]

import numpy as np


### cho bai 1
def grad(w):
    x = w[0].copy()
    y = w[1].copy()    
    return np.asarray([2*x, 2*y])

def cost(w):
    x = w[0].copy()
    y = w[1].copy()
    return x**2 + y**2


### cho bai 2
# def grad(w):
#     x = w[0].copy()
#     y = w[1].copy()
#     fx = 4*x*(x**2 + y - 7) + 2*(x - y + 1)
#     fy = 2*(x**2 + y - 7) - 2*(x - y + 1)
#     return np.asarray([fx, fy])

# def cost(w):
#     x = w[0].copy()
#     y = w[1].copy()
#     return (x**2 + y - 7)**2 + (x - y + 1)**2


import check_grad
print(check_grad.check_grad(np.random.rand(2, 1), cost, grad))

def GD(w0, ete = 0.1):
    w = [w0]
    for _ in range(1000):
        new_w = w[-1] - ete*grad(w[-1])
        w.append(new_w)
        if np.linalg.norm(grad(new_w), 2) / len(new_w) < 1e-3:
            break
    return w[-1]

w0 = np.random.rand(2, 1)
w = GD(w0)
print(w)



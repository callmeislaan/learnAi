# f(x) = x^2 + 5sin(x)
# f'(x) = 2x + 5cos(x)

import numpy as np

N = 10000

def grad(x):
    return 2*x + 5*np.cos(x)

def cost(x):
    return x**2 + 5*np.sin(x)

def myGD(x0, eta = 1e-3):
    x = [x0]
    for i in range(N):
        new_x = x[-1] - eta*grad(x[-1])
        if abs(grad(new_x)) < 1e-5:
            break
        x.append(new_x)
    return x, i

x0 = np.random.rand()
x, i = myGD(x0)
print(x[-1])
print(i)
print(cost(x[-1]))
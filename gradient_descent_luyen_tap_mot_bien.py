### su dung voi ham y = 2x^2 + 3^x + 4 dao ham y' = 4x + 3
import numpy as np
import matplotlib.pyplot as plt

# tinh dao ham
def grad(x):
    return 4 * x + 3

def cost(x):
    return 2*x**2 + 3*x + 4

def myGD(x0, ete):
    x = [x0]
    for i in range(100):
        x_new = x[-1] - ete * grad(x[-1])
        x.append(x_new)
        if abs(x_new) < 1e-6:
            break
    return x, i
# x, i = myGD(10, 0.1)
# print(i)
# print(x[-1])
# print(cost(x[-1]))
# print(grad(x[-1]))
# plt.plot(x[-1], cost(x[-1]), 'ro')
# a = np.arange(-10, 10, 0.1)
# b = cost(a)
# plt.plot(a, b)
# plt.show()
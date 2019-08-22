import numpy as np
# f(x) = x^2 + 10*sin(x)
# f'(x) = 2*x + 10*cos(x)

def grad(x):
    return 2*x + 10*np.cos(x)

def cost(x):
    return x**2 + 10*np.sin(x)

def has_converged(thena_new, grad):
    return np.linalg.norm(grad(thena_new)) / len(thena_new) < 1e-3

# thena_init: diem khoi tao(diem dat bi ban dau)
# gamma: van toc truoc do cua diem (hon bi) gia su ban dau = 0

def GD_momentum(thena_init, grad ,eta = 0.1, gamma = 0):
    thena = [thena_init]
    v_old = np.zeros_like(thena_init)
    for _ in range(100):
        v_new = gamma*v_old + eta*grad(thena[-1])
        thena_new = thena[-1] - v_new
        if has_converged(thena_new, grad):
            break
        thena.append(thena_new)
        v_old = v_new
    return thena[-1]

thena = GD_momentum(np.random.rand(1), grad)
print(thena)
import matplotlib.pyplot as plt
x = np.arange(-5, 5, 0.1)
y = cost(x)
plt.plot(x, y)
plt.plot(thena, cost(thena), 'ro')
plt.show()
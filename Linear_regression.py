import numpy as np
from sklearn import linear_model
import matplotlib.pyplot

# height(cm)
X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
# weight (kg)
y = np.array([ 49, 50, 51, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68])

one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis = 1)

print(Xbar)

A = Xbar.T.dot(Xbar)
B = Xbar.T.dot(y)
w = np.linalg.pinv(A).dot(B)
w_0 = w[0]
w_1 = w[1]
print(w)
y1 = w_1*155 + w_0
print(y1)

# use sklearn
w = linear_model.LinearRegression().fit(X, y)
print(w.coef_)
print(w.intercept_)

import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

## load data
breast_cancer = datasets.load_breast_cancer()
X = breast_cancer['data']
y = breast_cancer['target']
X_train, X_test, y_train, y_test = train_test_split(X, y)

## logistic regression
def sigmoid(s):
    return 1 / (1 + np.exp(-s))

def logistic_regression(w_init, X, y, eta = 1e-5, max_count = 10000):
    w = [w_init]
    check_w_after = 20
    count = 0
    N = X.shape[0]
    while count < max_count:
        mix_id = np.random.permutation(N)
        for i in mix_id:
            xi = X[i, :].reshape(1, -1)
            yi = y[i]
            zi = sigmoid(w[-1].dot(xi.T))
            w_new = w[-1] + (yi - zi) * xi
            count += 1
            if count%check_w_after == 0:
                if np.linalg.norm(w_new - w[-check_w_after]) < 1e-5:
                    return w
            
            w.append(w_new)
    return w

w_init = np.random.randn(1, X.shape[1])
w = logistic_regression(w_init, X_train, y_train)
y_pred = (sigmoid(w[-1].dot(X_test.T))).T
# print(y_pred.shape)
# print(y_test.shape)
print(accuracy_score(y_test, y_pred)*100)

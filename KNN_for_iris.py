import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load data
iris = datasets.load_iris()
X = iris['data']
y = iris['target']

X_train, X_test, y_train, y_test = train_test_split(X, y)

def dist_fast(Z, X):
    Z2 = np.sum(Z*Z, axis = 1)
    X2 = np.sum(X*X, axis = 1)
    return Z2.reshape(-1, 1) + X2.reshape(1, -1) - 2*Z.dot(X.T)

def find_min(res):
    rs = np.zeros(X_test.shape[0])
    for i in range(X_test.shape[0]):
        rs[i] = y_train[np.argmin(res[i])]
    return rs

y_pred = find_min(dist_fast(X_test, X_train)) 
print(y_pred)
print(y_test)
print(accuracy_score(y_test, y_pred)*100)

## use sklearn

Knei = KNeighborsClassifier(n_neighbors=5, p=2, weights='distance').fit(X_train, y_train)
y_pred = Knei.predict(X_test)
print(accuracy_score(y_test, y_pred)*100)
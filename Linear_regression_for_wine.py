import numpy as np
from sklearn import datasets, model_selection, metrics

wine = datasets.load_wine()
X = wine['data']
y = wine['target'].reshape(-1, 1)
one = np.ones((X.shape[0], 1))
X = np.concatenate((one, X), axis = 1)

n_train = int(0.8*X.shape[0])
X_train = X[:n_train, :]
X_test = X[n_train:, :]
y_train = y[:n_train, :]
y_test = y[n_train:, :]

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y)

A = X_train.T.dot(X_train)
B = X_train.T.dot(y_train)
w = np.linalg.pinv(A).dot(B)
print(w)

y_pred = np.round(X_test.dot(w), decimals = 0)
print(y_pred)

print(metrics.accuracy_score(y_test, y_pred)*100)
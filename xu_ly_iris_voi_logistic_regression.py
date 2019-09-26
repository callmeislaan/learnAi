import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# import some data to play with
# iris = datasets.load_iris()
# X = iris['data']
# y = iris['target']
# X_train, X_test, y_train, y_test = train_test_split(X, y)

# logreg = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')
# logreg.fit(X_train, y_train)
# y_pred = logreg.predict(X_test)
# print(accuracy_score(y_test, y_pred)*100)

# wine = datasets.load_wine()
# X = wine['data']
# y = wine['target']
# X_train, X_test, y_train, y_test = train_test_split(X, y)

# logreg = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')
# logreg.fit(X_train, y_train)
# y_pred = logreg.predict(X_test)
# print(accuracy_score(y_test, y_pred)*100)


## play with breast cancer dataset
dt = datasets.load_breast_cancer()
X = dt['data']
y = dt['target']
X_train, X_test, y_train, y_test = train_test_split(X, y)

logreg = LogisticRegression(C=1e5).fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print(accuracy_score(y_test, y_pred)*100)
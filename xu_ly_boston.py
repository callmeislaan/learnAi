import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

boston = datasets.load_boston()
X_train, X_test, y_train, y_test = train_test_split(boston['data'], boston['target'])



# use Kneighbors
# knei = KNeighborsClassifier(n_neighbors=10).fit(X_train, y_train)
# y_pred1 = knei.predict(X_test)

# print(y_pred1)
# print('ti le: ', accuracy_score(y_test, y_pred1)*100)

# use linearRegression
regr = LinearRegression(fit_intercept=False).fit(X_train, y_train)
y_pred2 = np.round(regr.predict(X_test), decimals=1)
print(y_test[:10])
print(y_pred2[:10])

plt.plot(y_test)
plt.plot(y_pred2)
plt.show()
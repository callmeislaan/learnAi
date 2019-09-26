import numpy as np
from sklearn import datasets, model_selection, neighbors, metrics
import matplotlib.pyplot as plt

# load data
boston = datasets.load_boston()
X_train, X_test, y_train, y_test = model_selection.train_test_split(boston['data'], boston['target'])
print(X_train.shape)

Kneig = neighbors.KNeighborsRegressor(n_neighbors=1).fit(X_train, y_train)
y_pred = np.round(Kneig.predict(X_test), decimals = 1)
print(y_pred[:5])
print(y_test[:5])
plt.plot(y_test, 'red')
plt.plot(y_pred)
plt.show()
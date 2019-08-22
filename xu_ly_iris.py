import numpy as np
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression

iris = datasets.load_iris()
X = iris['data']
y = iris['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

print(X_train.shape)
print(X_test.shape)

kmeans = KMeans(n_clusters=3, random_state=0).fit(X_train, y_train)
y_pred = kmeans.predict(X_test)

print(y_test)
print(y_pred)

print('ti le: ', accuracy_score(y_test, y_pred)*100)

knei = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
y_pred1 = knei.predict(X_test)
print('ti le 1: ', accuracy_score(y_test, y_pred1)*100)

regr = LinearRegression()
regr.fit(X_train,y_train)
y_pred2 = np.round(regr.predict(X_test))
# print(y_test)
# print(y_pred)
print('ti le 2:', accuracy_score(y_test, y_pred2)*100)
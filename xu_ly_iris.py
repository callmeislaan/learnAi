import numpy as np
from sklearn import datasets

iris = datasets.load_iris()
X = iris['data']
y = iris['target']

print(X.shape)
print(y.shape)
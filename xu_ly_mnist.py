import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os
import gzip
from sklearn.linear_model import LinearRegression

# load data
def load_mnist(file_path, kind = 'train'):
    images_path = os.path.join(file_path, '%s-images-idx3-ubyte.gz' % kind)
    lables_path = os.path.join(file_path, '%s-labels-idx1-ubyte.gz' % kind)

    with gzip.open(lables_path, 'rb') as lbl_path:
        labels = np.frombuffer(lbl_path.read(), dtype=np.uint8, offset=8)
    
    with gzip.open(images_path, 'rb') as img_path:
        images = np.frombuffer(img_path.read(), dtype=np.uint8, offset=16).reshape(len(labels), 28*28)

    return images, labels

X_train, y_train = load_mnist('Data/mnist')
X_test, y_test = load_mnist('Data/mnist', 't10k')

# display 100 random data
# fig = plt.figure(figsize=(9, 9))
# rows = cols = 10
# for i in range(1, rows*cols + 1):
#     img = X_train[np.random.randint(0, len(X_train))].reshape(28, 28)
#     fig.add_subplot(rows, cols, i)
#     plt.imshow(img)
#     plt.axis('off')

# plt.show()

# use kmeans
# kmeans = KMeans(n_clusters=10).fit(X_train, y_train)
# y_pred = kmeans.predict(X_test)
# centers = kmeans.cluster_centers_
# print('y_pred: ', y_pred[:10])
# print('y_test: ', y_test[:10])

# print('ti le: ', accuracy_score(y_test, y_pred)*100)

# show center and cluster
# rows = cols = 10
# fig = plt.figure(figsize=(10, 10))
# for i in range(centers.shape[0]):
#     img = centers[i].reshape(28, 28)
#     fig.add_subplot(rows, cols, i+1)
#     plt.imshow(img)
#     plt.axis('off')
# plt.show()

# user LinerRegression
# regr = LinearRegression(fit_intercept=False).fit(X_train, y_train)
# y_pred = np.round(regr.predict(X_test), decimals=0).astype(int)
# print(y_pred[:10])
# print(y_test[:10])

# print('accuracy: ', accuracy_score(y_test, y_pred)*100)

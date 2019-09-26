# on tap kmeans cluster voi bai toan phan phan chia so 0 1 va 2
import numpy as np
from scipy.spatial.distance import cdist
import os
import gzip
import matplotlib.pyplot as plt

## load MNIST
def load_MNIST(path, kind = 'train'):
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' %kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' %kind)

    with gzip.open(labels_path, mode = 'rb') as lbl:
        labels = np.frombuffer(lbl.read(), dtype = np.uint8, offset = 8)
    
    with gzip.open(images_path, mode = 'rb') as img:
        images = np.frombuffer(img.read(), dtype = np.uint8, offset = 16).reshape(-1, 28*28)
    
    return images, labels

X, y = load_MNIST('Data/mnist/')

X0 = X[y == 0, :]
X1 = X[y == 1, :]
X2 = X[y == 2, :]
X = np.concatenate((X0, X1, X2), axis = 0)
y = np.array([0]*X0.shape[0] + [1]*X1.shape[0] + [2]*X2.shape[0])

## kmeans cluster
def create_center_init(X, K):
    return X[np.random.choice(X.shape[0], K, replace=False)]

def update_labels(X, K, centers):
    dist = cdist(X, centers)
    label = np.argmin(dist, axis = 1)
    return label

def update_centers(X, K, labels):
    centers = np.array(np.zeros((K, X.shape[1])))
    for k in range(K):
        centers[k] = np.mean(X[labels == k, :], axis = 0)
    return centers

def has_converged(centers, new_centers):
    return set([tuple(c) for c in centers]) == set([tuple(c) for c in new_centers])

def kmeans(X, K):
    centers = [create_center_init(X, K)]
    while True:
        labels = update_labels(X, K, centers[-1])
        new_centers = update_centers(X, K, labels)
        if has_converged(centers[-1], new_centers):
            break
        centers.append(new_centers)
    return centers[-1]

K = 3
c = kmeans(X, K).reshape(-1, 28, 28)
plt.subplot(131)
plt.imshow(c[0])
plt.subplot(132)
plt.imshow(c[1])
plt.subplot(133)
plt.imshow(c[2])
plt.show()
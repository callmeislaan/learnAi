import numpy as np
import gzip
import os
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

# load data
def load_mnist(path, kind = 'train'):
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)

    with gzip.open(labels_path, mode='rb') as lblpath:
        labels = np.frombuffer(lblpath.read(), dtype=np.uint8, offset=8)
    
    with gzip.open(images_path, mode='rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(-1, 28*28)
    
    return images, labels

images, labels = load_mnist('Data/mnist/')

def display_image(rows, cols):
    fig = plt.figure(figsize=(9, 9))
    for i in range(1, rows*cols+1):
        img = images[i].reshape(28, 28)
        fig.add_subplot(rows, cols, i)
        plt.imshow(img)
        plt.axis('off')
    plt.show()

rows = cols = 10
# display_image(rows, cols)    

# init centroids
def kmeans_init_centroids(X, K):
    return X[np.random.choice(X.shape[0], size = K, replace = False)]

# assign labels
def kmeans_assign_labels(X, centroids):
    D = cdist(X, centroids)
    return np.argmin(D, axis = 1)

# update centroids
def kmeans_update_centroids(X, labels, K):
    centroids = np.zeros(shape = (K, X.shape[1]))
    for k in range(K):
        Xk = X[labels == k, :]
        centroids[k, :] = np.mean(Xk, axis=0)
    return centroids

# check converged
def has_converged(old_centroids, new_centroids):
    return set([tuple(a) for a in old_centroids]) == set([tuple(b) for b in new_centroids])

# kmeans main
def kmeans(X, K):
    centroids = [kmeans_init_centroids(X, K)]
    labels = []
    while True:
        labels = kmeans_assign_labels(X, centroids[-1])
        new_centroids = kmeans_update_centroids(X, labels, K)
        if has_converged(centroids[-1], new_centroids):
            break
        centroids.append(new_centroids)
    return centroids, labels

# img = images[labels == 1][0].reshape(28, 28)
# plt.imshow(img)
# plt.show()

# centroids, labels = kmeans(images, 10)
kmeans = KMeans(n_clusters=10).fit(images)
labels = kmeans.predict(images)
centroids = kmeans.cluster_centers_
def display_centroids(centroids):
    fig = plt.figure(figsize = (10, 10))
    i = 1
    k = 0
    while i <= 10*10:
        img = centroids[k, :].reshape(28, 28)
        fig.add_subplot(10, 10, i)
        plt.imshow(img)
        plt.axis('off')
        for j in range(1, 10):
            img = images[labels == k][j].reshape(28, 28)
            fig.add_subplot(10, 10, i)
            plt.imshow(img)
            plt.axis('off')
            i += 1
            if i%10 == 0:
                k += 1
        print(k)
    plt.show()

display_centroids(centroids)
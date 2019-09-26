import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist

# load data
iris = datasets.load_iris()
X = iris['data']
y = iris['target']

def kmeans_init_centroids(X, K):
    return X[np.random.choice(X.shape[0], K, replace=False)]

def kmeans_assign_labels(X, centroids):
    D = cdist(X, centroids)
    return np.argmin(D, axis = 1)

def kmeans_update_centroids(X, labels, K):
    centroids = np.zeros((K, X.shape[1]))
    for k in range(K):
        Xk = X[labels == k]
        centroids[k, :] = np.mean(Xk, axis = 0)
    return centroids

def has_converged(old_centroids, new_centroids):
    return set([tuple(a) for a in old_centroids]) == set([tuple(a) for a in new_centroids])

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

centroids, labels = kmeans(X, 3)
print(labels)
print(y)
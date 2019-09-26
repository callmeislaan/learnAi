import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

means = np.array([[1, 2], [3, 5], [4, 1]])
cov = np.array([[1, 0], [0, 1]])
N = 500
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)

X = np.concatenate((X0, X1, X2), axis = 0)
K = 3
label = np.asarray([0]*N + [1]*N + [2]*N)

def display_clusters(X, label):
    X0 = X[label == 0, :]
    X1 = X[label == 1, :]
    X2 = X[label == 2, :]
    plt.plot(X0[:, 0], X0[:, 1], 'go', markersize = 4)
    plt.plot(X1[:, 0], X1[:, 1], 'rs', markersize = 4)
    plt.plot(X2[:, 0], X2[:, 1], 'b^', markersize = 4)
    # plt.show()

# display_clusters(X, label)

# chon centroids ban dau
def kmeans_init_centroids(X, K):
    return X[np.random.choice(X.shape[0], size=K, replace=False)]

# gan labels cac diem du lieu
def kmeans_assign_labels(X, centroids):
    D = cdist(X, centroids)
    return np.argmin(D, axis = 1)

# cap nhat centroids
def kmeans_update_centroids(X, labels, K):
    centroids = []
    for k in range(K):
        Xk = X[labels == k, :]
        centroids.append(np.mean(Xk, axis = 0))
    return centroids

# kiem tra dieu kien dung
def has_converged(centroids, new_centroids):
    old_cen = set([tuple(a) for a in centroids])
    new_cen = set([tuple(a) for a in new_centroids])
    return old_cen == new_cen

# kmeans
def kmeans(X, K):
    centroids = [kmeans_init_centroids(X, K)]
    labels = []
    it = 0
    while True:
        labels.append(kmeans_assign_labels(X, centroids[-1]))
        new_centroids = kmeans_update_centroids(X, labels[-1], K)
        if has_converged(centroids[-1], new_centroids):
            break
        centroids.append(new_centroids)
        it += 1
    return (centroids, labels, it)

centroids, labels, it = kmeans(X, K)
print('centroids: ', centroids[-1])
print('iteration: ', it)
print('labels: ', set(labels[-1]))
plt.subplot(1, 2, 1)
display_clusters(X, labels[-1])

km = KMeans(n_clusters = K, random_state = 0).fit(X)
print(km.cluster_centers_)
labels_predict = km.predict(X)
plt.subplot(1, 2, 2)
display_clusters(X, labels_predict)
plt.show()
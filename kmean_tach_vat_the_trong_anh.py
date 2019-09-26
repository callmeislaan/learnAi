import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import matplotlib.pyplot as plt

# load image
img = plt.imread('Data/images/cogai.jpg')
X = img.reshape(img.shape[0]*img.shape[1], img.shape[2])

K = 4
kmeans = KMeans(n_clusters=K).fit(X)
centroids = kmeans.cluster_centers_
labels = kmeans.predict(X)
img2 = np.array(np.zeros_like(X))
for k in range(K):
    img2[labels == k] = centroids[k]

img2 = img2.reshape((img.shape[0], img.shape[1], img.shape[2]))
Image.fromarray(img2).save('Data/images/cogai1.jpg')
plt.imshow(img2)
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import accuracy_score
import os
import gzip

def load_MNIST(path, kind = 'train'):
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz'%kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte.gz'%kind)

    with gzip.open(labels_path) as lbl:
        lables = np.frombuffer(lbl.read(), dtype=np.uint8, offset=8)
    
    with gzip.open(images_path) as img:
        images = np.frombuffer(img.read(), dtype=np.uint8, offset=16).reshape(-1, 28*28)
    
    return lables, images

y_train, X_train = load_MNIST('Data/mnist/')
y_test, X_test = load_MNIST('Data/mnist/', kind='t10k')

logreg = linear_model.LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
print(accuracy_score(y_test, y_pred.tolist())*100)
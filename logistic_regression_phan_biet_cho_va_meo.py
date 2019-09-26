import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
from PIL import Image
from time import time
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB

img_size = 50

def rgb2gray(rgb):
    return rgb[:,:,:,0]*.299 + rgb[:,:, :, 1]*.587 + rgb[:,:, :, 2]*.114

# load dogs-vs-cats dataset
def load_data():
    X_train_path = 'Data/dogs-vs-cats/X_train.npy'
    X_test_path = 'Data/dogs-vs-cats/X_test.npy'
    y_train_path = 'Data/dogs-vs-cats/y_train.npy'
    y_test_path = 'Data/dogs-vs-cats/y_test.npy'
    X_train = np.load(X_train_path)
    X_test = np.load(X_test_path)
    y_train = np.load(y_train_path)
    y_test = np.load(y_test_path)

    X_train = rgb2gray(X_train).reshape(-1, 50*50)
    X_test = rgb2gray(X_test).reshape(-1, 50*50)

    return X_train, X_test, y_train, y_test

# giam chieu du lieu xuong con 500
def scale_small_feature_vector(matrix, d = 500):
    fat_matrix = np.random.randn(matrix.shape[1], d)
    return matrix.dot(fat_matrix)

def feature_extraction(X):
    x_mean = np.mean(X, axis = 0)
    x_var = np.var(X, axis = 0)
    return (X - x_mean) / x_var

X_train, X_test, y_train, y_test = load_data()
X_train = feature_extraction(scale_small_feature_vector(X_train, d = 1000))
X_test = feature_extraction(scale_small_feature_vector(X_test, d = 1000))

# t1 = time()
# logistic = LogisticRegression(C=1e5).fit(X_train, y_train)
# y_pred = logistic.predict(X_test)
# print(accuracy_score(y_test, y_pred)*100)
# print('time: ', time() - t1)

t1 = time()
per = Perceptron().fit(X_train, y_train)
y_pred = per.predict(X_test)
print(accuracy_score(y_test, y_pred)*100)
print('time: ', time() - t1)

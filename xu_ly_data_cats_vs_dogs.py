import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm
import os

# set labels
def label_img(img):
    word_label = img.split('.')[0]
    if word_label == 'cat':
        return 0
    else:
        return 1
    
def create_cats_and_dogs_datasets(train_path, img_size = 50):
    X = []
    y = []
    for img in tqdm(os.listdir(train_path)):
        label = label_img(img)
        path = os.path.join(train_path, img)
        img = Image.open(path)
        img = img.resize((img_size, img_size), Image.ANTIALIAS)
        img = np.asarray(img)
        X.append(img)
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    np.save('Data/dogs-vs-cats/X_test.npy', X)
    np.save('Data/dogs-vs-cats/y_test.npy', y)

    return X, y

X, y = create_cats_and_dogs_datasets('Data/dogs-vs-cats/test1')
print(X.shape)
print(y.shape)


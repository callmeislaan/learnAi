import numpy as np
from sklearn.naive_bayes import MultinomialNB

# create data
d1 = [2, 1, 1, 0, 0, 0, 0, 0, 0]
d2 = [1, 1, 0, 1, 1, 0, 0, 0, 0]
d3 = [0, 1, 0, 0, 1, 1, 0, 0, 0]
d4 = [0, 1, 0, 0, 0, 0, 1, 1, 1]
labels = np.array(['B', 'B', 'B', 'N'])
X =  np.asarray([d1, d2, d3, d4])

# training
V = X.shape[1]
NB = np.sum(X[labels == 'B', :]) + V
NN = np.sum(X[labels == 'N', :]) + V

pB = X[labels == 'B'].shape[0] / X.shape[0]
pN = X[labels == 'N'].shape[0] / X.shape[0]

lamdaB = []
lamdaN = []
for i in range(X.shape[1]):    
    lamdaB.append((np.sum(X[labels == 'B', i]) + 1) / NB)
    lamdaN.append((np.sum(X[labels == 'N', i]) + 1) / NN)

lamdaB = np.asarray(lamdaB)
lamdaN = np.asarray(lamdaN)

d5 = np.asarray([[2, 0, 0, 1, 0, 0, 0, 1, 0]])
d6 = np.asarray([[0, 1, 0, 0, 0, 0, 0, 1, 1]])
pBd5 = np.prod([a**b for a, b in zip(lamdaB, d5[0])])*pB
pNd5 = np.prod([a**b for a, b in zip(lamdaN, d5[0])])*pN
pBd6 = np.prod([a**b for a, b in zip(lamdaB, d6[0])])*pB
pNd6 = np.prod([a**b for a, b in zip(lamdaN, d6[0])])*pN

# show result
s = pBd5 + pNd5
print((pBd5 / s) * 100)
print((pNd5 / s) * 100)
s1 = pBd6 + pNd6
print()
print((pBd6 / s1) * 100)
print((pNd6 / s1) * 100)

# use bliblaly
MNB = MultinomialNB().fit(X, labels)
print('predict of class: ', str(MNB.predict(d5)[0]))
print('d6 probability of d6: ', MNB.predict_proba(d6)[0])
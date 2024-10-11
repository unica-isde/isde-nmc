import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from splitters import split_data
from classifiers import NMC


data = pd.read_csv("data/mnist_data.csv")
data = np.array(data)
print(data.shape)
print(type(data))

y = data[:, 0]
x = data[:, 1:] / 255

idx = 100
plt.imshow(x[idx, :].reshape(28, 28), cmap="gray")
plt.show()
print(y[idx])

xtr, ytr, xts, yts = split_data(x, y, tr_fraction=0.6)
print(xtr.shape, ytr.shape)

xk = xtr[ytr == 0, :]  # all images of zeros
plt.figure()
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(xk[i, :].reshape(28, 28), cmap='gray')
plt.show()
print(xk.shape)

meank = np.mean(xk, axis=0)
plt.figure()
plt.imshow(meank.reshape(28, 28), cmap='gray')
plt.show()

# create an instance of the NMC classifier
clf = NMC()
clf.fit(xtr, ytr)

centroids = clf.centroids

clf.centroids = 1

plt.figure()
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(centroids[i, :].reshape(28, 28), cmap='gray')
plt.show()

print(centroids.shape)
print(xts.shape)

ypred = clf.predict(xts)

accuracy = np.mean(ypred == yts)
print("Accuracy:", accuracy)




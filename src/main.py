import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from splitters import split_data
from classifiers import NMC
from loaders import DataLoaderMNIST, DataLoaderLFW

data_loader = DataLoaderLFW()
x, y = data_loader.load_data()

idx = 100
plt.imshow(x[idx, :].reshape(data_loader.w, data_loader.h), cmap="gray")
plt.show()
print(y[idx])

xtr, ytr, xts, yts = split_data(x, y, tr_fraction=0.6)
print(xtr.shape, ytr.shape)

plt.figure()
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x[i, :].reshape(data_loader.w, data_loader.h), cmap='gray')
plt.show()
print(x.shape)

# create an instance of the NMC classifier
clf = NMC()
clf.fit(xtr, ytr)

centroids = clf.centroids

plt.figure()
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(centroids[i, :].reshape(
        data_loader.w, data_loader.h), cmap='gray')
plt.show()

print(centroids.shape)
print(xts.shape)

ypred = clf.predict(xts)

accuracy = np.mean(ypred == yts)
print("Accuracy:", accuracy)

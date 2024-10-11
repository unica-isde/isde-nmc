import numpy as np
from sklearn.metrics import pairwise_distances


def softmax(x):
    """Compute softmax values for each row in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)


class NMC():
    def __init__(self):
        self.centroids = None

    @property
    def centroids(self):
        return self._centroids

    # @centroids.setter
    # def centroids(self, value):
    #    self._centroids = value

    def fit(self, xtr, ytr):
        """ Compute the average centroid for each class."""

        n_dimensions = xtr.shape[1]
        n_classes = np.unique(ytr).size
        self._centroids = np.zeros(shape=(n_classes, n_dimensions))
        for k in range(n_classes):
            # extract images from one class and then average along dim 0
            self._centroids[k, :] = np.mean(xtr[ytr == k, :], axis=0)
        return self

    def decision_function(self, xts):
        """
        Compute similarities with centroids
        :param xts:
        :return:
        """
        if self.centroids is None:
            raise ValueError(
                "Centroids have not been estimated. Call `fit' first.")

        dist = pairwise_distances(xts, self.centroids)
        sim = 1 / (1e-3 + dist)
        return sim

    def predict(self, xts):
        scores = self.decision_function(xts)
        ypred = np.argmax(scores, axis=1)
        return ypred

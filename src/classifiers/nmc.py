import numpy as np
from sklearn.metrics import pairwise_distances


def softmax(x):
    """
    Softmax-scaling of input matrix values.
    Each row will be normalized to sum up to one.

    Parameters
    ----------
    x : ndarray
        the input matrix

    Returns
    -------
        the softmax-scaled outputs
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)


class NMC:
    """
    Class defined for NMC classifier.
    """

    def __init__(self):
        self._centroids = None

    @property
    def centroids(self):
        return self._centroids

    # @centroids.setter
    # def centroids(self, value):
    #    self._centroids = value

    def fit(self, xtr, ytr):
        """
        Compute the average centroids for each class

        Parameters
        ----------
        xtr: training data
        ytr: training labels

        Returns
        -------
        self: trained NMC classifier
        """

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

        Parameters
        ----------
        xts : ndarray
            Input samples to be classified

        Returns
        -------
            Output values for each sample vs class
        """
        if self.centroids is None:
            raise ValueError(
                "Centroids have not been estimated. Call `fit' first.")

        dist = pairwise_distances(xts, self.centroids)
        sim = 1 / (1e-3 + dist)
        return sim


    def predict(self, xts):
        """
        Brand new docstring

        Parameters
        ----------
        xts

        Returns
        -------

        """
        scores = self.decision_function(xts)
        ypred = np.argmax(scores, axis=1)
        return ypred

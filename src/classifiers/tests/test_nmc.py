import unittest

import numpy as np
from classifiers import NMC


class TestNMC(unittest.TestCase):

    def setUp(self):
        self.n_samples = 10
        self.n_features = 100
        self.n_classes = 2
        self.x = np.zeros(shape=(self.n_samples, self.n_features))
        self.x[-1, :] = 1
        self.y = np.zeros(shape=(self.n_samples))
        self.y[-1] = 1
        self.clf = NMC()

    def test_fit(self):
        self.clf.fit(self.x, self.y)
        self.assertEqual(
            (self.n_classes, self.n_features), self.clf.centroids.shape)
        centroids = np.zeros(shape=(self.n_classes, self.n_features))
        centroids[-1, :] = 1
        self.assertEqual(0,
            np.round(np.mean(centroids-self.clf.centroids), 6))

    def test_predict(self):
        pass



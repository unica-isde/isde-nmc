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
        # test the exception
        self.assertRaises(ValueError, self.clf.predict, self.x)

        self.clf.fit(self.x, self.y)
        ypred = self.clf.predict(self.x)
        self.assertEqual(ypred.shape, self.y.shape)
        self.assertEqual(np.sum(ypred==self.y), self.n_samples)

        probs = self.clf.decision_function(self.x, softmax_scaling=True)
        v = np.sum(probs, axis=1)
        self.assertAlmostEqual(self.n_samples, np.sum(v))



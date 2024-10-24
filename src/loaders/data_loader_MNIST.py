import numpy as np
import pandas as pd
from loaders import DataLoader


class DataLoaderMNIST(DataLoader):
    def __init__(self, scaling=True):
        self.scaling = scaling
        self._w = 28
        self._h = 28

    @property
    def w(self):
        return self._w

    @property
    def h(self):
        return self._h

    @property
    def scaling(self):
        return self._scaling

    @scaling.setter
    def scaling(self, scaling):
        self._scaling = bool(scaling)

    def load_data(self):
        data = pd.read_csv("data/mnist_data.csv")
        data = np.array(data)

        y = data[:, 0]
        x = data[:, 1:]
        if self.scaling:
            x = x / 255.0

        return x, y

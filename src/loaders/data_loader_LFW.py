from loaders import DataLoader
from sklearn.datasets import fetch_lfw_people

class DataLoaderLFW(DataLoader):

    def __init__(self):
        self._w, self._h = 62, 47

    @property
    def w(self):
        return self._w

    @property
    def h(self):
        return self._h

    def load_data(self):
        x, y = fetch_lfw_people(return_X_y=True, min_faces_per_person=10)
        return x, y
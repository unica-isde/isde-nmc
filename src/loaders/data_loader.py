from abc import abstractmethod, ABC


class DataLoader(ABC):

    @abstractmethod
    def load_data(self):
        raise NotImplementedError("The load_data method must be implemented.")

from abc import ABC, abstractmethod

class ModelFromK(ABC):
    @abstractmethod
    def __init__(self, *args):
        raise NotImplementedError
    @abstractmethod
    def get_loss(self, *args):
        raise NotImplementedError
    @abstractmethod
    def get_gradient(self, *args):
        raise NotImplementedError


from abc import ABC, abstractmethod

class ModelFromK(ABC):
    @abstractmethod
    def __init__(self, *args):
        raise NotImplementedError
    @abstractmethod
    def getLoss(self, *args):
        raise NotImplementedError
    @abstractmethod
    def getGradient(self, *args):
        raise NotImplementedError


from abc import ABC, abstractmethod

class ModelFromK(ABC):
    @abstractmethod
    def __init__(self, p:int, c:int, l:int, structure_list, op_pool):
        raise NotImplementedError
    @abstractmethod
    def getLoss(self, super_circ_params):
        raise NotImplementedError
    @abstractmethod
    def getGradient(self, super_circ_params):
        raise NotImplementedError


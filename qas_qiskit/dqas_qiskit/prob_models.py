import numpy as np
import jax.numpy as jnp
from abc import ABC, abstractmethod

class ProbabilityModelBaseClass(ABC):

    @abstractmethod
    def get_loss(self):
        pass

    @abstractmethod
    def get_gradient(self):
        pass



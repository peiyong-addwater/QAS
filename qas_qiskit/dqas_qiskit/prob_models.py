import jax.nn
import numpy as np
import jax.ops
import jax.numpy as jnp
from abc import ABC, abstractmethod
import optax
from typing import (
    List,
    Sequence,
    Any,
    Tuple,
    Callable,
    Iterator,
    Optional,
    Union,
    Iterable,
    Dict,
    AnyStr
)

class ProbModelBaseClass(ABC):

    @abstractmethod
    def get_gradient(self, *args):
        pass

    @abstractmethod
    def get_prob_matrix(self):
        pass

    @abstractmethod
    def get_parameters(self):
        pass



class IndependentCategoricalProbabilisticModel(ProbModelBaseClass):
    def __init__(self, alpha:jnp.ndarray):
        self.c = alpha.shape[1] # c columns, pool size
        self.alpha = alpha
        self.p = alpha.shape[0] # p rows, num of layers
        self.prob_matrix = jax.nn.softmax(self.alpha, axis=1)

    def get_gradient(self, circ_loss:float, k:List[int]):
        assert max(k)<self.c
        assert min(k)>=0
        assert len(k) == self.p
        nabla_ln_P = jnp.zeros((self.p, self.c))
        for i in range(len(k)):
            m = k[i]
            for j in range(self.alpha.shape[1]):
                update = self.delta_symbol(j, m) - self.prob_matrix[i,m]
                nabla_ln_P = jax.ops.index_update(nabla_ln_P, (i,j),update)
        return nabla_ln_P*circ_loss

    def delta_symbol(self, j:int, m:int)->int:
        return int(j==m)

    def get_prob_matrix(self):
        return self.prob_matrix

    def get_parameters(self):
        return self.alpha

def categorical_sample(alpha, key):
    sampled = jax.random.categorical(key, alpha, axis=1)
    return [int(c) for c in sampled]






"""
alpha = jnp.random.randn(12).reshape(3,4)
start_learning_rate = 1e-1
optimizer = optax.adam(start_learning_rate)
opt_state = optimizer.init(alpha)
loss = list(np.arange(1,0,-0.1))
loss.append(0)
print(loss)
for i in range(len(loss)):
    print("epoch {}".format(i+1))
    l = loss[i]
    k = categorical_sample(alpha, jax.random.PRNGKey(0))
    prob_model = IndependentCategoricalProbabilisticModel(alpha)
    grads = prob_model.get_gradient(l, k)
    updates, opt_state = optimizer.update(grads, opt_state)
    alpha = optax.apply_updates(alpha, updates)
    print(alpha)
"""



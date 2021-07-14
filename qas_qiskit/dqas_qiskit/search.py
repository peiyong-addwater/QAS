import numpy as np
import jax.numpy as jnp
import jax
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
from prob_models import ProbModelBaseClass, IndependentCategoricalProbabilisticModel, categorical_sample
from circuits import QCircFromK, BitFlipSearchDensityMatrix, SIMPLE_DATASET_BIT_FLIP
from standard_ops import GatePool, default_complete_graph_parameterized_pool

def dqas(num_epochs:int, init_prob_params:np.ndarray, init_circ_params:np.ndarray, op_pool:GatePool, search_circ:QCircFromK,
         prob_model:ProbModelBaseClass=IndependentCategoricalProbabilisticModel, circ_opt = optax.adam,
         prob_opt = optax.adam, ):
    pass
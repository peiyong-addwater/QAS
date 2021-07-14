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
from prob_models import ProbModelBaseClass, IndependentCategoricalProbabilisticModel
from circuits import QCircFromK, BitFlipSearchDensityMatrix, SIMPLE_DATASET_BIT_FLIP
from standard_ops import GatePool, default_complete_graph_parameterized_pool


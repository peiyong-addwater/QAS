import numpy as np
import jax.numpy as jnp
import jax
import time
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
from dqas_qiskit.prob_models import (
    ProbModelBaseClass,
    IndependentCategoricalProbabilisticModel,
    categorical_sample)
from dqas_qiskit.circuits import (
    QCircFromK,
    BitFlipSearchDensityMatrixNoiseless,
    SIMPLE_DATASET_BIT_FLIP,
    FiveBitCodeSearchDensityMatrixNoiseless,
    SIMPLE_DATASET_FIVE_BIT_CODE
)
from dqas_qiskit.standard_ops import (
    GatePool,
    default_complete_graph_parameterized_pool,
    default_complete_graph_non_parameterized_pool
)
from dqas_qiskit.search import train_circuit, dqas_qiskit

pool =default_complete_graph_parameterized_pool(5)
p = 30
c = len(pool)
l = 3
param = np.random.randn(p*c*l).reshape((p,c,l))
a = np.zeros(p*c)
a = a.reshape((p,c))
final_prob_param, final_circ_param, final_prob_model, final_circ, final_k, final_op_list, final_loss= dqas_qiskit(
    50, SIMPLE_DATASET_BIT_FLIP, a, param, pool, FiveBitCodeSearchDensityMatrixNoiseless,
    IndependentCategoricalProbabilisticModel,
    prob_train_k_num_samples=200, verbose=2,train_circ_in_between_epochs=10,parameterized_circuit=True
)
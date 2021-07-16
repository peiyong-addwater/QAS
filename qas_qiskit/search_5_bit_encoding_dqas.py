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



line_five_qubits_connection = [(0,1),(1,0),(1,2),(2,1),(2,3),(3,2),(3,4),(4,3)]
single_qubit_gate = ["U3Gate"]
two_qubit_gate = ["CU3Gate"]


#pool =default_complete_graph_parameterized_pool(5)
pool = GatePool(5, single_qubit_gate, two_qubit_gate, False, line_five_qubits_connection)
print(pool)
p = 30
c = len(pool)
l = 3
param = np.random.randn(p*c*l).reshape((p,c,l))
a = np.zeros(p*c).reshape((p,c))
final_prob_param, final_circ_param, final_prob_model, final_circ, final_k, final_op_list, final_loss, loss_list_qas= dqas_qiskit(
    500, SIMPLE_DATASET_FIVE_BIT_CODE, a, param, pool, FiveBitCodeSearchDensityMatrixNoiseless,
    IndependentCategoricalProbabilisticModel,
    prob_train_k_num_samples=300, verbose=2,train_circ_in_between_epochs=10,parameterized_circuit=True
)

"""
# Fine tune the circuit after architecture search
final_circ_param, final_circ, final_op_list, _ = train_circuit(50, FiveBitCodeSearchDensityMatrixNoiseless,
                                                               final_circ_param, final_k, pool,
                                                               SIMPLE_DATASET_FIVE_BIT_CODE, lr= 0.1,
                                                               verbose=2, early_stopping_threshold=0.000001)
"""

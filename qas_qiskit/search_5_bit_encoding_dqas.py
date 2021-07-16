import numpy as np
import jax.numpy as jnp
import jax
import time
import json
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

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def nowtime():
    return str(time.strftime("%Y%m%d-%H%M", time.localtime()))

file_name = nowtime()+"_FIVE_BIT_CODE_SEARCH.json"
res_dict = {}

line_five_qubits_connection = [(0,1),(1,0),(1,2),(2,1),(2,3),(3,2),(3,4),(4,3)]
single_qubit_gate = ["U3Gate"]
two_qubit_gate = ["CU3Gate"]


pool =default_complete_graph_parameterized_pool(5)
#pool = GatePool(5, single_qubit_gate, two_qubit_gate, False, line_five_qubits_connection)

res_dict["Pool"] = str(pool)

print(pool)
p = 30
c = len(pool)
l = 3

res_dict["Search_Param"] = {"p":p, "c":c, "l":l}

param = np.random.randn(p*c*l).reshape((p,c,l))
a = np.zeros(p*c).reshape((p,c))
final_prob_param, final_circ_param, final_prob_model, final_circ, final_k, final_op_list, final_loss, loss_list_qas= dqas_qiskit(
    500, SIMPLE_DATASET_FIVE_BIT_CODE, a, param, pool, FiveBitCodeSearchDensityMatrixNoiseless,
    IndependentCategoricalProbabilisticModel,
    prob_train_k_num_samples=300, verbose=2,train_circ_in_between_epochs=20,parameterized_circuit=True
)
res_dict["k"] = final_k
res_dict["op_list"] = final_op_list
res_dict["loss_list"] = loss_list_qas
res_dict["prob_param"] = final_prob_param
res_dict["all_circuit_param"] = final_circ_param


with open(file_name, 'w') as f:
    json.dump(res_dict, f, indent=4, cls=NpEncoder)




# Fine tune the circuit after architecture search
tuned_circ_param, tuned_circ, tuned_op_list, _, fine_tune_loss_list = train_circuit(500, FiveBitCodeSearchDensityMatrixNoiseless,
                                                               np.random.randn(p*c*l).reshape((p,c,l)), final_k, pool,
                                                               SIMPLE_DATASET_FIVE_BIT_CODE, lr= 0.01,
                                                               verbose=2, early_stopping_threshold=0.000001)

res_dict["fine_tune_res"] = {
    "tuned_circ_param":tuned_circ_param,
    "tuned_op_list":tuned_op_list,
    "fine_tune_loss_list":fine_tune_loss_list
}

with open(file_name, 'w') as f:
    json.dump(res_dict, f, indent=4, cls=NpEncoder)
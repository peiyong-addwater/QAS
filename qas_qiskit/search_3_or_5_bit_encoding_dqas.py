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
from dqas_qiskit.search import train_circuit, dqas_qiskit, dqas_qiskit_v2, dqas_qiskit_v2_weighted_gradients

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
if __name__ == "__main__":

    file_name = nowtime()+"_QEC_CODE_SEARCH.json"
    restricted_pool = True

    num_qubits= 5
    if num_qubits !=3:
        assert num_qubits == 5

    res_dict = {}
    res_dict["NUM_QUBITS"] = num_qubits

    line_five_qubits_connection = [(0,1),(1,0),(1,2),(2,1),(2,3),(3,2),(3,4),(4,3)]
    single_qubit_gate = ["U3Gate"]
    two_qubit_gate = ["CU3Gate"]


    #pool =default_complete_graph_parameterized_pool(num_qubits)
    if num_qubits == 5 and restricted_pool:
        pool = GatePool(5, single_qubit_gate, two_qubit_gate, False, line_five_qubits_connection)
        file_name = nowtime() + "_QEC_CODE_SEARCH_RESTRICTED_POOL.json"
    else:
        pool = default_complete_graph_parameterized_pool(num_qubits)

    res_dict["Pool"] = str(pool)

    print(pool)
    p = 3 if num_qubits ==3 else 30
    c = len(pool)
    l = 3

    res_dict["Search_Param"] = {"p":p, "c":c, "l":l}

    param = np.random.randn(p*c*l).reshape((p,c,l))
    a = np.zeros(p*c).reshape((p,c))
    # add some hints for prob parameters:
    # a[0,0], a[0,1], a[0,2], a[0,3], a[0,4], a[0,5], a[0,6], a[0,7], a[0,8] = 1, 1, 1, 1, 1, 1, 1, 1, 1


    final_prob_param, final_circ_param, final_prob_model, final_circ, final_k, final_op_list, final_loss, loss_list_qas=\
        dqas_qiskit_v2_weighted_gradients(num_epochs=1000,
                    training_data=SIMPLE_DATASET_BIT_FLIP if num_qubits==3 else SIMPLE_DATASET_FIVE_BIT_CODE,
                    init_prob_params=a,
                    init_circ_params=param,
                    op_pool=pool,
                    search_circ_constructor=BitFlipSearchDensityMatrixNoiseless if num_qubits == 3 else FiveBitCodeSearchDensityMatrixNoiseless,
                    circ_lr=0.1,
                    prob_lr = 0.1,
                    circ_opt = optax.adabelief,
                    prob_opt = optax.adabelief,
                    prob_model=IndependentCategoricalProbabilisticModel,
                    batch_k_num_samples=300,
                    verbose=2,
                    parameterized_circuit=True
    )
    res_dict["k"] = final_k
    res_dict["op_list"] = final_op_list
    res_dict["loss_list"] = loss_list_qas
    res_dict["prob_param"] = final_prob_param
    res_dict["all_circuit_param"] = final_circ_param




    # Fine tune the circuit after architecture search
    tuned_circ_param, tuned_circ, tuned_op_list, _, fine_tune_loss_list = \
        train_circuit(1000,
                    circ_constructor=FiveBitCodeSearchDensityMatrixNoiseless if num_qubits==5 else BitFlipSearchDensityMatrixNoiseless,
                    init_params=np.random.randn(p*c*l).reshape((p,c,l)),
                    k=final_k,
                    op_pool=pool,
                    training_data=SIMPLE_DATASET_FIVE_BIT_CODE if num_qubits==5 else SIMPLE_DATASET_BIT_FLIP,
                    lr= 0.1,
                    verbose=2,
                    early_stopping_threshold=0.000001,
                    opt=optax.adabelief
                    )

    res_dict["fine_tune_res"] = {
        "tuned_circ_param":tuned_circ_param,
        "tuned_op_list":tuned_op_list,
        "fine_tune_loss_list":fine_tune_loss_list
    }

    with open(file_name, 'w') as f:
        json.dump(res_dict, f, indent=4, cls=NpEncoder)
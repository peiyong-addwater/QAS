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
    PhaseFlipDensityMatrixNoiseless,
    SIMPLE_DATASET_BIT_FLIP,
    FiveBitCodeSearchDensityMatrixNoiseless,
    SIMPLE_DATASET_FIVE_BIT_CODE,
    SIMPLE_DATASET_PHASE_FLIP,
    FOUR_TWO_TWO_DETECTION_CODE_DATA,
    FourTwoTwoDetectionDensityMatrixNoiseless
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
    return str(time.strftime("%Y%m%d-%H%M%S", time.localtime()))
if __name__ == "__main__":

    task = 'FOUR_TWO_TWO_DETECTION'

    TASKS = ['BIT_FLIP', 'PHASE_FLIP', 'FOUR_TWO_TWO_DETECTION','FIVE_BIT_CODE']
    assert task in TASKS

    file_name = nowtime()+"_"+task+"_QEC_CODE_SEARCH.json"
    restricted_pool = False

    if task == "BIT_FLIP":
        batch_k_num_samples = 300
        num_qubits= 3
        p = 2
        circ_constructor = BitFlipSearchDensityMatrixNoiseless
        data_set = SIMPLE_DATASET_BIT_FLIP
        prob_noise_factor = 1/50
        circ_grad_noise_factor=0
        last_20_opt_circ_loss_std_threshold =0
        local_opt_trapped_max_count = 10,
        batch_loss_threshold = 0.01
        pool = default_complete_graph_parameterized_pool(num_qubits)
        num_epochs = 100
        force_escape_prob_local_min = False
        c = len(pool)
        l = 3
        a = np.zeros(p * c).reshape((p, c))
    elif task == 'PHASE_FLIP':
        batch_k_num_samples = 300
        num_qubits = 3
        p=3
        circ_constructor = PhaseFlipDensityMatrixNoiseless
        data_set = SIMPLE_DATASET_PHASE_FLIP
        prob_noise_factor = 1/50
        circ_grad_noise_factor = 0
        last_20_opt_circ_loss_std_threshold =0
        local_opt_trapped_max_count = 10,
        batch_loss_threshold = 0.01
        pool = default_complete_graph_parameterized_pool(num_qubits)
        num_epochs = 200
        force_escape_prob_local_min = False
        c = len(pool)
        l = 3
        a = np.zeros(p * c).reshape((p, c))
    elif task == 'FOUR_TWO_TWO_DETECTION':
        batch_k_num_samples = 400
        num_qubits = 4
        p=6
        data_set = FOUR_TWO_TWO_DETECTION_CODE_DATA
        circ_constructor = FourTwoTwoDetectionDensityMatrixNoiseless
        prob_noise_factor = 1/10
        circ_grad_noise_factor = 1/20
        last_20_opt_circ_loss_std_threshold =0.05
        local_opt_trapped_max_count = 10,
        batch_loss_threshold = 0.01
        pool = default_complete_graph_parameterized_pool(num_qubits)
        num_epochs = 500
        force_escape_prob_local_min = True
        c = len(pool)
        l = 3
        a = np.zeros(p * c).reshape((p, c))
        if restricted_pool:
            connection = [(0,2), (2,0), (0,3), (3,0), (1, 2), (2, 1), (1,3), (3,1), (2, 3), (3, 2)]
            single_qubit_gate = ["U3Gate"]
            two_qubit_gate = ["CU3Gate"]
            pool = GatePool(4, single_qubit_gate, two_qubit_gate, False, connection)
            file_name = nowtime() + "422_DETECTION_CODE_SEARCH_RESTRICTED_POOL.json"
            c = len(pool)
            l = 3
            a = np.zeros(p * c).reshape((p, c))
    elif task == 'FIVE_BIT_CODE':
        batch_k_num_samples = 1200
        num_epochs = 500
        num_qubits = 5
        p=18
        circ_constructor = FiveBitCodeSearchDensityMatrixNoiseless
        date_set =SIMPLE_DATASET_FIVE_BIT_CODE
        prob_noise_factor = 1/10
        circ_grad_noise_factor = 0
        last_20_opt_circ_loss_std_threshold =0
        force_escape_prob_local_min = False
        local_opt_trapped_max_count = 10,
        batch_loss_threshold = 0.01
        pool = default_complete_graph_parameterized_pool(num_qubits)
        c = len(pool)
        l = 3
        a = np.zeros(p * c).reshape((p, c))
        if restricted_pool:
            line_five_qubits_connection = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2), (3, 4), (4, 3)]
            single_qubit_gate = ["U3Gate"]
            two_qubit_gate = ["CU3Gate"]
            pool = GatePool(5, single_qubit_gate, two_qubit_gate, False, line_five_qubits_connection)
            file_name = nowtime() + "FIVE_BIT_QEC_CODE_SEARCH_RESTRICTED_POOL.json"
            c = len(pool)
            l = 3
            a = np.zeros(p * c).reshape((p, c))
    else:
        force_escape_prob_local_min = False
        local_opt_trapped_max_count = 10,
        batch_loss_threshold = 0.01
        batch_k_num_samples = 300
        num_epochs = 500
        num_qubits = 0
        circ_grad_noise_factor = 0
        last_20_opt_circ_loss_std_threshold =0
        p=0
        circ_constructor = None
        data_set = None
        prob_noise_factor = 1/50
        pool = default_complete_graph_parameterized_pool(num_qubits)
        c = len(pool)
        l = 3
        a = np.zeros(p * c).reshape((p, c))
        exit(-1)


    res_dict = {}
    res_dict["NUM_QUBITS"] = num_qubits

    res_dict["Pool"] = str(pool)
    print(pool)

    res_dict["Search_Param"] = {"p":p, "c":c, "l":l}


    param = np.random.randn(p*c*l).reshape((p,c,l))
    res_dict['circ_param_init'] = param
    res_dict['prob_param_init'] = a

    #TODO: Add beam search for structure parameters
    final_prob_param, final_circ_param, final_prob_model, final_circ, final_k, final_op_list, final_loss, \
    loss_list_qas, loss_std, prob_params_list=\
        dqas_qiskit(num_epochs=num_epochs,
                    training_data=data_set,
                    init_prob_params=a,
                    init_circ_params=param,
                    op_pool=pool,
                    search_circ_constructor=circ_constructor,
                    circ_lr= 0.1,
                    prob_lr = 0.1,
                    circ_opt = optax.adabelief,
                    prob_opt = optax.adabelief,
                    prob_model=IndependentCategoricalProbabilisticModel,
                    batch_k_num_samples=batch_k_num_samples,
                    verbose=12,
                    parameterized_circuit=True,
                    prob_grad_noise_factor=prob_noise_factor,
                    circ_grad_noise_factor=circ_grad_noise_factor,
                    force_escape_prob_local_min=force_escape_prob_local_min,
                    last_20_opt_circ_loss_std_threshold=last_20_opt_circ_loss_std_threshold,
                    local_opt_trapped_max_count=local_opt_trapped_max_count,
                    batch_loss_threshold=batch_loss_threshold
                    )
    res_dict["k"] = final_k
    res_dict["op_list"] = final_op_list
    res_dict["loss_list"] = loss_list_qas
    res_dict["prob_params_list"] = prob_params_list
    res_dict["loss_std"] = loss_std
    res_dict["final_prob_param"] = final_prob_param
    res_dict["all_circuit_param"] = final_circ_param






    # Fine tune the circuit after architecture search
    tuned_circ_param, tuned_circ, tuned_op_list, _, fine_tune_loss_list = \
        train_circuit(500,
                    circ_constructor=circ_constructor,
                    init_params=np.random.randn(p*c*l).reshape((p,c,l)),
                    k=final_k,
                    op_pool=pool,
                    training_data=data_set,
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
from cmab_qas.search import searchParameterized, single_circuit_training
from cmab_qas.standard_ops import GatePool
from cmab_qas.circuits import (
    BitFlipSearchDensityMatrixNoiseless,
    SIMPLE_DATASET_BIT_FLIP,
    PhaseFlipDensityMatrixNoiseless,
    SIMPLE_DATASET_PHASE_FLIP,
    FourTwoTwoDetectionDensityMatrixNoiseless,
    FOUR_TWO_TWO_DETECTION_CODE_DATA,
    FiveBitCodeSearchDensityMatrixNoiseless,
    SIMPLE_DATASET_FIVE_BIT_CODE,
    TOFFOLI_DATA,
    ToffoliCircuitDensityMatrixNoiseless
)
import json
import numpy as np
import optax
import jax.numpy as jnp
import jax
import time
import random

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

    # random.seed(2077) # Thessia

    marker = nowtime()
    filename = marker+'.json'
    task = "FIVE_BIT_CODE"
    model = FiveBitCodeSearchDensityMatrixNoiseless
    data = SIMPLE_DATASET_FIVE_BIT_CODE
    init_qubit_with_actions = {0}
    d_np = ["CU3Gate"]
    s_np = ["U3Gate"]
    pool = GatePool(5, s_np, d_np, complete_undirected_graph=True, two_qubit_gate_map=None)
    p = 16
    l = 3
    c = len(pool)


    init_params = np.random.randn(p,c,l)
    final_params, final_best_arc, final_best_node, final_best_reward, final_controller = searchParameterized(
        model=model,
        data=data,
        init_qubit_with_actions=init_qubit_with_actions,
        init_params=init_params,
        num_iterations=500,
        num_warmup_iterations=50,
        iteration_limit=5,
        arc_batchsize=100,
        alpha_max=2,
        alpha_min=1/np.sqrt(2),
        prune_constant_min=0.5,
        prune_constant_max=0.9,
        eval_expansion_factor=100,
        op_pool=pool,
        target_circuit_depth=p,
        first_policy='local_optimal',
        second_policy='local_optimal',
        super_circ_train_iterations=10,
        super_circ_train_optimizer=optax.adam,
        super_circ_train_gradient_noise_factor=1/20,
        super_circ_train_lr=0.01,
        iteration_limit_ratio=5,
        num_minimum_children=10,
        checkpoint_file_name_start=marker
    )

    # train the best arc:
    final_params, loss_list = single_circuit_training(
        initial_params=final_params,
        circ_constructor=model,
        num_epochs=200,
        k = final_best_arc,
        op_pool=pool,
        training_data=data,
        optimizer_callable=optax.adam,
        lr=0.01,
        grad_noise_factor=1/20,
        verbose=1
    )



    res_dict = {
        'task':task,
        'pool':pool.pool,
        'params':final_params,
        'k':final_best_arc,
        'ops':model(p,c,l,final_best_arc,pool).get_circuit_ops(final_params),
        'final_training_res':{
            'params':final_params,
            'loss_list':loss_list
        }
    }
    with open(filename, 'w') as f:
        json.dump(res_dict, f, indent=4, cls=NpEncoder)

from cmab_qas.search import searchParameterized, single_circuit_training, TreeNode
from cmab_qas.standard_ops import GatePool
from cmab_qas.circuits import *
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


    marker = nowtime()
    filename = marker+'.json'
    task = "TOFFOLI_RESTRICTED_POOL_6_CX_PENALTY_DP_NOISE"
    # task = "422_FULL_CONNECTION_CX_U3"
    model = ToffoliCircuitDMNoisy
    # model = FourTwoTwoDetectionDensityMatrixNoiseless
    data = TOFFOLI_DATA
    # data = FOUR_TWO_TWO_DETECTION_CODE_DATA
    init_qubit_with_actions = {0,1,2}
    d_np = ["CXGate"]
    s_np = ["U3Gate"]
    cu3_map = [(0,1), (0,2), (1,2)]
    pool = GatePool(3, s_np, d_np, complete_undirected_graph=False, two_qubit_gate_map=cu3_map)
    #pool = GatePool(4, s_np, d_np)
    p = 20
    l = 3
    c = len(pool)

    # penalty function:
    def penalty_func(r:float, node:TreeNode):
        k = node.state.current_k
        cu3_count = 0
        for op_index in k:
            op_name = list(pool[op_index].keys())[0]
            if "CU3" in op_name or "CX" in op_name or "CR" in op_name or "CY" in op_name or "CZ" in op_name:
                cu3_count = cu3_count + 1
        if cu3_count>=10:
            return r - (cu3_count-11)
        return r




    init_params = np.random.randn(p,c,l)
    final_params, final_best_arc, final_best_node, final_best_reward, final_controller, reward_list = searchParameterized(
        model=model,
        data=data,
        init_qubit_with_actions=init_qubit_with_actions,
        init_params=init_params,
        num_iterations=500,
        num_warmup_iterations=100,
        iteration_limit=10,
        arc_batchsize=50,
        alpha_max=2,
        alpha_min=1/np.sqrt(2),
        prune_constant_min=0.5,
        prune_constant_max=0.95,
        eval_expansion_factor=100,
        op_pool=pool,
        target_circuit_depth=p,
        first_policy='local_optimal',
        second_policy='local_optimal',
        super_circ_train_optimizer=optax.adam,
        super_circ_train_gradient_noise_factor=1/40,
        super_circ_train_lr=0.01,
        iteration_limit_ratio=5,
        num_minimum_children=10,
        checkpoint_file_name_start=marker,
        penalty_function=penalty_func
    )

    # train the best arc:
    final_params, loss_list = single_circuit_training(
        initial_params=final_params,
        circ_constructor=model,
        num_epochs=500,
        k = final_best_arc,
        op_pool=pool,
        training_data=data,
        optimizer_callable=optax.adam,
        lr=0.01,
        grad_noise_factor=1/100,
        verbose=1
    )



    res_dict = {
        'task':task,
        'pool':pool.pool,
        'params':final_params,
        'k':final_best_arc,
        'ops':model(p,c,l,final_best_arc,pool).get_circuit_ops(final_params),
        "reward_list":reward_list,
        'final_training_res':{
            'params':final_params,
            'loss_list':loss_list,
            'ops':model(p,c,l,final_best_arc,pool).get_circuit_ops(final_params)
        }
    }
    with open(filename, 'w') as f:
        json.dump(res_dict, f, indent=4, cls=NpEncoder)

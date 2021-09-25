from qas.mcts import search, TreeNode, circuitModelTuning
from qas.qml_ops import QMLPool
from qas.qml_models import ToffoliQMLNoiseless, PhaseFlipQMLNoiseless
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
    task = "TOFFOLI_RESTRICTED_POOL_RZ_X_SX_CNOT"
    model = ToffoliQMLNoiseless
    init_qubit_with_actions = {0,1,2}
    two_qubit_gate = ["CNOT"]
    single_qubit_gate = ["SX", "RZ"]
    #single_qubit_gate = ['U3']
    control_map = [[0,1], [1,2], [1,0], [2,1]]
    pool = QMLPool(3, single_qubit_gate, two_qubit_gate, complete_undirected_graph=False, two_qubit_gate_map=control_map)
    print(pool)
    p = 37
    l = 3
    c = len(pool)

    # penalty function:
    def penalty_func(r:float, node:TreeNode):
        k = node.state.getCurrK()
        cu3_count = 0
        for op_index in k:
            op_name = list(pool[op_index].keys())[0]
            if "CNOT" in op_name or  "CR" in op_name or "CY" in op_name or "CZ" in op_name or "CRot" in op_name:
                cu3_count = cu3_count + 1
        if cu3_count>=8:
            return r - (cu3_count-8)
        return r


    init_params = np.random.randn(p,c,l)


    final_params, final_best_arc, final_best_node, final_best_reward, final_controller, reward_list = search(
        model=model,
        op_pool=pool,
        target_circuit_depth=p,
        init_qubit_with_controls=init_qubit_with_actions,
        init_params=init_params,
        num_iterations=400,
        num_warmup_iterations=100,
        super_circ_train_optimizer=optax.adam,
        super_circ_train_gradient_noise_factor=1/50,
        super_circ_train_lr=0.1,
        penalty_function=penalty_func,
        arc_batchsize=200,
        alpha_max=2,
        alpha_min=1/np.sqrt(2),
        prune_constant_max=0.6,
        prune_constant_min=0.5,
        max_visits_prune_threshold=100,
        min_num_children=5,
        sampling_execute_rounds=20,
        exploit_execute_rounds=5,
        sample_policy='local_optimal',
        exploit_policy='local_optimal'
    )


    final_params, loss_list = circuitModelTuning(
        initial_params=final_params,
        model=model,
        num_epochs=100,
        k=final_best_arc,
        op_pool=pool,
        opt_callable=optax.adam,
        lr=0.01,
        grad_noise_factor=0,
        verbose=1
    )

    res_dict = {
        'task':task,
        'pool':pool.pool,
        'params':final_params,
        'k':final_best_arc,
        'op_list':model(p,c,l,final_best_arc,pool).toList(final_params),
        'search_reward_list':reward_list,
        'tuning_loss_list':loss_list
    }

    with open(filename, 'w') as f:
        json.dump(res_dict, f, indent=4, cls=NpEncoder)



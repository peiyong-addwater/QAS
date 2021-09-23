from qas.mcts import search, TreeNode
from qas.qml_ops import QMLPool
from qas.qml_models import ToffoliQMLNoiseless
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
    task = "TOFFOLI_RESTRICTED_POOL_RZ_X_SX_CX"
    # task = "422_FULL_CONNECTION_CX_U3"
    model = ToffoliQMLNoiseless
    init_qubit_with_actions = {0,1,2}
    d_np = ["CNOT"]
    s_np = ["PauliX", "SX", "RZ"]
    cu3_map = [(0,1), (1,2)]
    pool = QMLPool(3, s_np, d_np, complete_undirected_graph=False, two_qubit_gate_map=cu3_map)
    print(pool)
    #pool = GatePool(4, s_np, d_np)
    p = 35
    l = 3
    c = len(pool)

    # penalty function:
    def penalty_func(r:float, node:TreeNode):
        k = node.state.getCurrK()
        cu3_count = 0
        for op_index in k:
            op_name = list(pool[op_index].keys())[0]
            if "CNOT" in op_name or  "CR" in op_name or "CY" in op_name or "CZ" in op_name:
                cu3_count = cu3_count + 1
        if cu3_count>=10:
            return r - (cu3_count-11)
        return r


    init_params = np.random.randn(p,c,l)


    final_params, final_best_arc, final_best_node, final_best_reward, final_controller, reward_list = search(
        model=model,
        op_pool=pool,
        target_circuit_depth=p,
        init_qubit_with_controls=init_qubit_with_actions,
        init_params=init_params,
        num_iterations=500,
        num_warmup_iterations=50,
        super_circ_train_optimizer=optax.adam,
        super_circ_train_gradient_noise_factor=1/100,
        super_circ_train_lr=0.1,
        penalty_function=penalty_func,
        arc_batchsize=200,
        alpha_max=2,
        alpha_min=1/np.sqrt(2),
        prune_constant_max=0.9,
        prune_constant_min=0.5,
        max_visits_prune_threshold=100,
        min_num_children=5,
        sampling_execute_rounds=100,
        exploit_execute_rounds=100,
        sample_policy='local_optimal',
        exploit_policy='local_optimal'
    )



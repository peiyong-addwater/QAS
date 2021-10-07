from qas.mcts import search, TreeNode, circuitModelTuning
from qas.qml_gate_ops import QMLPool
from qas.qml_models import FourTwoTwoQMLNoiseless
import json
import numpy as np
import pennylane as qml
import time
from qas.mcts import QMLStateBasicGates
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

    model = FourTwoTwoQMLNoiseless
    state_class = QMLStateBasicGates

    marker = nowtime()
    filename = marker + '.json'
    task = model.name + "_" + state_class.name
    print(task)
    init_qubit_with_actions = {0,1}
    two_qubit_gate = ["CNOT"]
    single_qubit_gate = ["U3", "PlaceHolder"]

    # set a hard limit on the number of certain gate instead of using a penalty function
    gate_limit = {"CNOT": 5}
    pool = QMLPool(4, single_qubit_gate, two_qubit_gate, complete_undirected_graph=True)
    print(pool)
    p = 6
    l = 3
    c = len(pool)
    ph_count_limit = 0


    # penalty function:
    def penalty_func(r: float, node: TreeNode):
        k = node.state.getCurrK()
        ph_count = 0
        for op_index in k:
            op_name = list(pool[op_index].keys())[0]
            if op_name == 'PlaceHolder':
                ph_count = ph_count + 1
        if ph_count >= ph_count_limit:
            return r - (ph_count - ph_count_limit) / 1
        return r


    init_params = np.random.randn(p, c, l)

    final_params, final_best_arc, final_best_node, final_best_reward, final_controller, reward_list = search(
        model=model,
        op_pool=pool,
        target_circuit_depth=p,
        init_qubit_with_controls=init_qubit_with_actions,
        init_params=init_params,
        num_iterations=50,
        num_warmup_iterations=10,
        super_circ_train_optimizer=qml.AdamOptimizer,
        super_circ_train_gradient_noise_factor=0,
        early_stop_threshold=0.95,
        early_stop_lookback_count=1,
        super_circ_train_lr=0.1,
        penalty_function=penalty_func,
        gate_limit_dict=gate_limit,
        warmup_arc_batchsize=200,
        search_arc_batchsize=100,
        alpha_max=3,
        alpha_min=1 / np.sqrt(2) / 2,
        prune_constant_max=0.99,
        prune_constant_min=0.70,
        max_visits_prune_threshold=50,
        min_num_children=3,
        sampling_execute_rounds=100,
        exploit_execute_rounds=100,
        cmab_sample_policy='local_optimal',
        cmab_exploit_policy='local_optimal',
        uct_sample_policy='local_optimal',
        verbose=2,
        state_class=state_class
    )

    final_params, loss_list = circuitModelTuning(
        initial_params=final_params,
        model=model,
        num_epochs=100,
        k=final_best_arc,
        op_pool=pool,
        opt_callable=qml.AdamOptimizer,
        lr=0.01,
        grad_noise_factor=0,
        verbose=1
    )

    res_dict = {
        'task': task,
        'pool': pool.pool,
        'params': final_params,
        'k': final_best_arc,
        'op_list': model(p, c, l, final_best_arc, pool).toList(final_params),
        'search_reward_list': reward_list,
        'fine_tune_loss': loss_list
    }

    with open(filename, 'w') as f:
        json.dump(res_dict, f, indent=4, cls=NpEncoder)

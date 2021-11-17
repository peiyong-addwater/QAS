from qas.mcts import search, TreeNode, circuitModelTuning
from qas.qml_gate_ops import QMLPool
from qas.qml_models import FourQubitH2
import json
import numpy as np
import pennylane as qml
import time
from qas.mcts import QMLStateBasicGates
import random
import warnings
#warnings.filterwarnings("ignore")

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

    import shutup
    shutup.please()

    model = FourQubitH2 #ground-state energy = -1.13618883 Ha
    state_class = QMLStateBasicGates


    marker = nowtime()
    filename = marker + '.json'
    task = model.name + "_" + state_class.name
    print(task)
    init_qubit_with_actions = set()
    two_qubit_gate = ["CNOT"]
    single_qubit_gate = ["Rot","PlaceHolder"]
    connection_graph = [[0,1],[1,0],[1,2],[2,1],[2,3],[3,2]]

    # set a hard limit on the number of certain gate instead of using a penalty function
    gate_limit = {"CNOT": 10}
    pool = QMLPool(4, single_qubit_gate, two_qubit_gate, complete_undirected_graph=False,
                   two_qubit_gate_map=connection_graph)
    print(pool)
    p = 20
    l = 3
    c = len(pool)
    ph_count_limit = p//2


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
        num_iterations=400,
        num_warmup_iterations=10,
        super_circ_train_optimizer=qml.AdamOptimizer,
        super_circ_train_gradient_noise_factor=0,
        early_stop_threshold=1.13,
        early_stop_lookback_count=1,
        super_circ_train_lr=0.5,
        penalty_function=penalty_func,
        gate_limit_dict=gate_limit,
        warmup_arc_batchsize=100,
        search_arc_batchsize=600,
        alpha_max=3,
        alpha_decay_rate=0.95,
        prune_constant_max=0.99,
        prune_constant_min=0.80,
        max_visits_prune_threshold=20,
        min_num_children=c // 4 + 1,
        sampling_execute_rounds=10,
        exploit_execute_rounds=50,
        cmab_sample_policy='local_optimal',
        cmab_exploit_policy='local_optimal',
        uct_sample_policy='local_optimal',
        verbose=2,
        state_class=state_class,
        search_reset=True
    )

    final_params, loss_list = circuitModelTuning(
        initial_params=final_params,
        model=model,
        num_epochs=400,
        k=final_best_arc,
        op_pool=pool,
        opt_callable=qml.AdamOptimizer,
        lr=0.1,
        grad_noise_factor=0,
        verbose=1,
        early_stop_threshold=-1.2
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

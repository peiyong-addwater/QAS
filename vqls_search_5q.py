from qas.mcts import search, TreeNode, circuitModelTuning
from qas.qml_gate_ops import QMLPool
from qas.qml_models import VQLSDemo5Q
import json
import numpy as np
import pennylane as qml
import time
from qas.mcts import QMLStateBasicGates
import random
import warnings
import matplotlib.pyplot as plt
plt.style.use(['science','nature'])
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

    model = VQLSDemo5Q
    state_class = QMLStateBasicGates
    num_qubits = 5

    q_delta = 0.001  # Initial spread of random quantum weights

    marker = nowtime()
    task = model.name + "_" + state_class.name
    filename = marker+'_' +task+'.json'
    print(task)
    init_qubit_with_actions = None
    two_qubit_gate = ["CNOT"]
    single_qubit_gate = ["Rot","PlaceHolder"]
    connection_graph = [[0,1],[1,2],[2,3],[3,4], [1,0], [2,1], [3,2], [4,3], [0,4], [4,0]]

    # set a hard limit on the number of certain gate instead of using a penalty function
    pool = QMLPool(num_qubits, single_qubit_gate, two_qubit_gate, complete_undirected_graph=False, two_qubit_gate_map=connection_graph)
    print(pool)
    p = 20
    l = 3
    c = len(pool)
    gate_limit = {"CNOT": p//2}
    ph_count_limit = 10

    # penalty function:
    def penalty_func(r: float, node: TreeNode):
        k = node.state.getCurrK()
        ph_count = 0
        for op_index in k:
            op_name = list(pool[op_index].keys())[0]
            if op_name == 'PlaceHolder':
                ph_count = ph_count + 1
        if ph_count >= ph_count_limit:
            return r - (ph_count - ph_count_limit) / 10
        return r

    init_params = np.random.randn(p, c, l)*q_delta

    final_params, final_best_arc, final_best_node, final_best_reward, final_controller, reward_list = search(
        model=model,
        op_pool=pool,
        target_circuit_depth=p,
        init_qubit_with_controls=init_qubit_with_actions,
        init_params=init_params,
        num_iterations=500,
        num_warmup_iterations=10,
        super_circ_train_optimizer=qml.GradientDescentOptimizer,
        super_circ_train_gradient_noise_factor=0.0,
        early_stop_threshold=0.9,
        early_stop_lookback_count=1,
        super_circ_train_lr=0.1,
        penalty_function=penalty_func,
        gate_limit_dict=gate_limit,
        warmup_arc_batchsize=500,
        search_arc_batchsize=20,
        alpha_max=3,
        alpha_decay_rate=0.99,
        prune_constant_max=0.70,
        prune_constant_min=0.30,
        max_visits_prune_threshold=10,
        min_num_children=c//2+1,
        sampling_execute_rounds=10,
        exploit_execute_rounds=c,
        cmab_sample_policy='local_optimal',
        cmab_exploit_policy='local_optimal',
        verbose=1,
        state_class=state_class,
        search_reset=False,
        avg_gradients=True
    )

    final_params, loss_list = circuitModelTuning(
        initial_params=final_params,
        model=model,
        num_epochs=500,
        k=final_best_arc,
        op_pool=pool,
        opt_callable=qml.AdamOptimizer,
        lr=0.01,
        grad_noise_factor=0,
        verbose=1,
        early_stop_threshold=-1
    )

    searched_model = model(p, c, l, final_best_arc, pool)
    quantum_result = searched_model.getQuantumSolution(final_params)
    classical_result = searched_model.getClassicalSolution()

    print("x_n^2 =\n", classical_result)
    print("|<x|n>|^2=\n", quantum_result)

    res_dict = {
        'task': task,
        'pool': pool.pool,
        'params': final_params,
        'k': final_best_arc,
        'op_list': model(p, c, l, final_best_arc, pool).toList(final_params),
        'search_reward_list': reward_list,
        'fine_tune_loss': loss_list,
        'quantum_result':quantum_result,
        'classical_result':classical_result
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.bar(np.arange(0, 2 ** num_qubits), classical_result, color="coral")
    ax1.set_xlim(-0.5, 2 ** num_qubits - 0.5)
    ax1.set_xlabel("Vector space basis")
    ax1.set_title("Classical probabilities")

    ax2.bar(np.arange(0, 2 ** num_qubits), quantum_result, color="lightblue")
    ax2.set_xlim(-0.5, 2 ** num_qubits - 0.5)
    ax2.set_xlabel("Hilbert space basis")
    ax2.set_title("Quantum probabilities")

    plt.savefig(marker + '_vqls_5q_search_demo.png')

    with open(filename, 'w') as f:
        json.dump(res_dict, f, indent=4, cls=NpEncoder)

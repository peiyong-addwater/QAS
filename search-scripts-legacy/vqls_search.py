from qas.mcts import search, TreeNode, circuitModelTuning
from qas.qml_models.qml_gate_ops import QMLPool
from qas.qml_models.qml_models_legacy import VQLSDemo
import json
import numpy as np
import pennylane as qml
import time
from qas.mcts import QMLStateBasicGates
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

    model = VQLSDemo
    state_class = QMLStateBasicGates
    num_qubits = 4


    marker = nowtime()
    task = model.name + "_" + state_class.name
    filename = marker+'_' +task+'.json'
    print(task)
    init_qubit_with_actions = None
    two_qubit_gate = ["CNOT"]
    single_qubit_gate = ["Rot","PlaceHolder"]
    connection_graph = [[0,1],[1,0],[1,2],[2,1],[2,3],[3,2],[0,3],[3,0]]

    # set a hard limit on the number of certain gate instead of using a penalty function
    pool = QMLPool(num_qubits, single_qubit_gate, two_qubit_gate, complete_undirected_graph=False, two_qubit_gate_map=connection_graph)
    print(pool)
    p = 10
    l = 3
    c = len(pool)
    gate_limit = {"CNOT": 8}
    ph_count_limit = 2

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

    init_params = np.random.randn(p, c, l)*np.sqrt(2/(2**num_qubits))

    final_params, final_best_arc, final_best_node, final_best_reward, final_controller, reward_list = search(
        model=model,
        op_pool=pool,
        target_circuit_depth=p,
        init_qubit_with_controls=init_qubit_with_actions,
        init_params=init_params,
        num_iterations=50,
        num_warmup_iterations=5,
        super_circ_train_optimizer=qml.GradientDescentOptimizer,
        super_circ_train_gradient_noise_factor=0,
        early_stop_threshold=0.9,
        early_stop_lookback_count=1,
        super_circ_train_lr=0.1,
        penalty_function=penalty_func,
        gate_limit_dict=gate_limit,
        warmup_arc_batchsize=50,
        search_arc_batchsize=40,
        alpha_max=1,
        alpha_decay_rate=0.9,
        prune_constant_max=0.90,
        prune_constant_min=0.60,
        max_visits_prune_threshold=5,
        min_num_children=4,
        sampling_execute_rounds=50,
        exploit_execute_rounds=100,
        cmab_sample_policy='local_optimal',
        cmab_exploit_policy='local_optimal',
        verbose=2,
        state_class=state_class,
        search_reset=True,
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

    plt.savefig(marker + '_vqls_search_4q_demo.pdf')

    with open(filename, 'w') as f:
        json.dump(res_dict, f, indent=4, cls=NpEncoder)

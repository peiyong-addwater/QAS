from qas.mcts import search, TreeNode, circuitModelTuning
from qas.qml_gate_ops import QMLPool
from qas.qml_models import QAOAVQCDemo
import json
import numpy as np
import pennylane as qml
import time
from qas.mcts import QMLStateBasicGates
import random
import warnings
import matplotlib.pyplot as plt
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

    model = QAOAVQCDemo
    state_class = QMLStateBasicGates
    num_qubits = 7


    marker = nowtime()
    task = model.name + "_" + state_class.name
    filename = marker+'_' +task+'.json'
    print(task)
    init_qubit_with_actions = None
    two_qubit_gate = ["CNOT"]
    single_qubit_gate = ["Rot","PlaceHolder"]
    connection_graph = [[0,1],[1,0],[1,2],[2,1],[2,3],[3,2],[3,4],[4,3], [4,5], [5,4], [5,6], [6,5], [6,0],[0,6]]

    # set a hard limit on the number of certain gate instead of using a penalty function
    pool = QMLPool(num_qubits, single_qubit_gate, two_qubit_gate, complete_undirected_graph=False, two_qubit_gate_map=connection_graph)
    print(pool)
    p = 15
    l = 3
    c = len(pool)
    ph_count_limit = 1
    cnot_count_soft_limit = p
    gate_limit = {"CNOT": p//2}

    init_params = np.random.randn(p, c, l)*np.sqrt(2/(2**num_qubits))

    final_params, final_best_arc, final_best_node, final_best_reward, final_controller, reward_list = search(
        model=model,
        op_pool=pool,
        target_circuit_depth=p,
        init_qubit_with_controls=init_qubit_with_actions,
        init_params=init_params,
        num_iterations=50,
        num_warmup_iterations=2,
        super_circ_train_optimizer=qml.AdamOptimizer,
        super_circ_train_gradient_noise_factor=0,
        early_stop_threshold=6.0,
        early_stop_lookback_count=1,
        super_circ_train_lr=0.1,
        penalty_function=None,
        gate_limit_dict=gate_limit,
        warmup_arc_batchsize=100,
        search_arc_batchsize=100,
        alpha_max=1,
        alpha_decay_rate=0.9,
        prune_constant_max=0.90,
        prune_constant_min=0.60,
        max_visits_prune_threshold=10,
        min_num_children=p//2,
        sampling_execute_rounds=100,
        exploit_execute_rounds=500,
        cmab_sample_policy='local_optimal',
        cmab_exploit_policy='local_optimal',
        verbose=1,
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
        early_stop_threshold=-999
    )

    searched_model = model(p, c, l, final_best_arc, pool)
    quantum_result = searched_model.getQuantumSolution(final_params)
    classical_result = searched_model.getClassicalSolution()

    print("Classical Res. ", classical_result)
    print("Quantum Res. ", quantum_result)

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

    with open(filename, 'w') as f:
        json.dump(res_dict, f, indent=4, cls=NpEncoder)

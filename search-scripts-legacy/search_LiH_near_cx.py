from qas.mcts import search, TreeNode, circuitModelTuning
from qas.qml_models.qml_gate_ops import QMLPool
from qas.qml_models.qml_models_legacy import LiH
import json
import numpy as np
import pennylane as qml
import time
from qas.mcts import QMLStateBasicGates


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

def generate_near_cx_connection_list(num_qubits:int):
    forward_list = []
    backward_list = []
    connection_list = []
    for i in range(num_qubits-1):
        fwd_connection = [i, i+1]
        bwd_connection = [i+1, i]
        connection_list.append(fwd_connection)
        connection_list.append(bwd_connection)
    return connection_list

if __name__ == "__main__":

    import shutup
    shutup.please()

    model = LiH
    # classical results (sto-6g)
    # converged SCF energy = -7.80324297491404
    # E(FCI) = -7.888567271126
    # converged SCF energy = -7.80324297491407  <S^2> = 3.5527137e-15  2S+1 = 1
    # E(UHF-FCI) = -7.888567271127
    # E(FCI) = -7.888567271126
    # adaptive circuit solution:
    # Total number of gates: 180
    # Total number of two-qubit control gates: 96
    state_class = QMLStateBasicGates

    target_energy = -7.888567271126


    marker = nowtime()
    task = model.name + "_" + state_class.name
    filename = marker + "_" + task + '.json'
    print(task)
    init_qubit_with_actions = set()
    two_qubit_gate = ["CNOT"]
    single_qubit_gate = ["Rot","PlaceHolder"]
    # set a hard limit on the number of certain gate instead of using a penalty function
    cx_connections = generate_near_cx_connection_list(10)
    pool = QMLPool(10, single_qubit_gate, two_qubit_gate, complete_undirected_graph=False, two_qubit_gate_map=cx_connections)
    print(pool)
    p = 20
    l = 3
    c = len(pool)
    ph_count_limit = p
    gate_limit = {"CNOT": p//2}


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
        num_warmup_iterations=2,
        super_circ_train_optimizer=qml.AdamOptimizer,
        super_circ_train_gradient_noise_factor=0.0,
        early_stop_threshold=999,
        early_stop_lookback_count=1,
        super_circ_train_lr=1,
        penalty_function=penalty_func,
        gate_limit_dict=gate_limit,
        warmup_arc_batchsize=500,
        search_arc_batchsize=25,
        alpha_max=2,
        alpha_decay_rate=0.99,
        prune_constant_max=0.9,
        prune_constant_min=0.8,
        max_visits_prune_threshold=10,
        min_num_children=5,
        sampling_execute_rounds=10,
        exploit_execute_rounds=20,
        cmab_sample_policy='local_optimal',
        cmab_exploit_policy='local_optimal',
        uct_sample_policy='local_optimal',
        verbose=1,
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
        early_stop_threshold=-99999
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

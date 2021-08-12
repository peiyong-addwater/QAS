import itertools
import time
import optax
from joblib import Parallel, delayed
from pprint import pprint
from typing import (
    List,
    Sequence,
    Any,
    Tuple,
    Callable,
    Iterator,
    Optional,
    Union,
    Iterable,
    Dict,
    AnyStr,
    Set
)
from .standard_ops import GatePool, default_complete_graph_parameterized_pool, default_complete_graph_non_parameterized_pool
from qiskit.quantum_info import DensityMatrix
from .tree_qas import MCTSController, QASState, TreeNode
import numpy as np
import jax
import jax.numpy as jnp


def _circ_obj_get_loss_dm(circ_obj, circ_params, init_state, target_state:List[DensityMatrix]):
    return circ_obj.get_loss(circ_params, init_state, target_state)

def _circ_obj_get_gradient_dm(circ_obj, circ_params, init_state, target_state:List[DensityMatrix]):
    return circ_obj.get_gradient(circ_params, init_state, target_state)


def searchNonParameterized(
        model,
        data:List[List],
        init_qubit_with_actions: Set,
        num_iterations = 500,
        num_warmup_iterations = 20,
        iteration_limit = 5,
        arc_batchsize = 20,
        alpha=1 / np.sqrt(2),
        prune_constant=0.9,
        eval_expansion_factor=100,
        op_pool: GatePool = None,
        target_circuit_depth=20,
        first_policy='local_optimal',
        second_policy='local_optimal'
):
    p = target_circuit_depth
    l = 3
    c = len(op_pool)
    assert len(data) == 2
    placeholder_params = np.zeros((p,c,l))
    assert num_warmup_iterations<num_iterations//2
    assert num_warmup_iterations >= 10
    controller = MCTSController(model,
                                data,
                                iteration_limit=iteration_limit,
                                alpha=alpha,
                                prune_constant=prune_constant,
                                eval_expansion_factor=eval_expansion_factor,
                                op_pool=op_pool,
                                init_qubit_with_actions=init_qubit_with_actions,
                                target_circuit_depth=target_circuit_depth,
                                first_policy=first_policy,
                                second_policy=second_policy)
    current_best_arc = None
    current_best_node = None
    current_best_reward = None
    controller.setRoot()
    pool_size = len(op_pool)

    for epoch in range(num_iterations):
        start = time.time()
        arcs, nodes = [], []
        if epoch<num_warmup_iterations:
            print("="*10+"Warming at Epoch {}\{}, Pool Size: {}".format(epoch+1, num_iterations, pool_size)+"="*10)
            for _ in range(arc_batchsize):
                k, node = controller.randomSample()
                arcs.append(k)
                nodes.append(node)
        else:
            print("=" * 10 + "Searching at Epoch {}\{}, Pool Size: {}".format(epoch + 1, num_iterations, pool_size) + "=" * 10)
            for _ in range(arc_batchsize):
                k, node = controller.sampleArc(placeholder_params)
                arcs.append(k)
                nodes.append(node)
        for k, node in zip(arcs, nodes):
            # TODO: Need parallel processing
            reward = controller.simulation(k, placeholder_params)
            controller.backPropagate(node, reward)
        current_best_arc, current_best_node = controller.exploitArc(placeholder_params)
        current_best_reward = controller.simulation(current_best_arc, placeholder_params)
        end = time.time()
        print("Prune Count: {}".format(controller.prune_counter))
        print("Current Best Reward: {}".format(current_best_reward))
        print("Current Best k:\n", current_best_arc)
        print("Current Ops:")
        print(current_best_node.state)
        print("="*10+"Epoch Time: {}".format(end-start)+"="*10)

    return current_best_arc, current_best_node, current_best_reward



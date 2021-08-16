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

def batch_training(initial_param_proxy:np.ndarray,
                   circ_constructor:Callable,
                   num_epochs:int,
                   k_list:List[List[int]],
                   op_pool:GatePool,
                   training_data:List[List],
                   optimizer_callable:Callable = optax.adam,
                   lr = 0.01,
                   circ_noise_factor = 1/20
                   ):
    p, c, l = initial_param_proxy.shape[0], initial_param_proxy.shape[1], initial_param_proxy.shape[2]
    batch_circs = [circ_constructor(p, c, l, k ,op_pool) for k in k_list]
    optimizer = optimizer_callable(lr)
    circ_params = initial_param_proxy
    opt_state = optimizer.init(circ_params)

    assert len(training_data) == 2
    input_state = training_data[0]
    target_state = training_data[1]
    for epoch in range(num_epochs):
        circ_batch_gradients = Parallel(n_jobs=-1, verbose=0)(
            delayed(_circ_obj_get_gradient_dm)(constructed_circ, circ_params, input_state, target_state)
            for constructed_circ in batch_circs
        )
        circ_batch_gradients = jnp.stack(circ_batch_gradients, axis=0)
        circ_batch_gradients = jnp.nan_to_num(circ_batch_gradients)
        circ_gradient = jnp.mean(circ_batch_gradients, axis=0)
        # add some noise to the circuit gradient
        seed = np.random.randint(0, 1000000000)
        key = jax.random.PRNGKey(seed)
        noise = jax.random.normal(key, shape=(p,c,l))
        circ_gradient = circ_gradient + noise*circ_noise_factor

        circ_updates, opt_state = optimizer.update(circ_gradient, opt_state)
        circ_params = optax.apply_updates(circ_params, circ_updates)
        circ_params = np.array(circ_params)

    final_params = circ_params
    final_loss_list = [circ_obj.get_loss(final_params, input_state, target_state) for circ_obj in batch_circs]
    return final_params, final_loss_list




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
        second_policy='local_optimal',
        iteration_limit_ratio = 10,
        num_minimum_children=10
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
                                second_policy=second_policy,
                                iteration_limit_ratio=iteration_limit_ratio,
                                num_minimum_children=num_minimum_children)
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


def searchParameterized(
        model,
        data:List[List],
        init_qubit_with_actions: Set,
        init_params:np.ndarray,
        num_iterations = 500,
        num_warmup_iterations = 20,
        iteration_limit = 5,
        arc_batchsize = 200,
        alpha=1 / np.sqrt(2),
        prune_constant=0.9,
        eval_expansion_factor=100,
        op_pool: GatePool = None,
        target_circuit_depth=20,
        first_policy='local_optimal',
        second_policy='local_optimal',
        super_circ_train_iterations = 20,
        super_circ_train_optimizer = optax.adam,
        super_circ_train_gradient_noise_factor = 1/20,
        super_circ_train_lr = 0.01,
        iteration_limit_ratio = 10,
        num_minimum_children=10
):
    p = target_circuit_depth
    l = init_params.shape[2]
    c = len(op_pool)
    assert len(data) == 2
    assert p == init_params.shape[0]
    assert c == init_params.shape[1]
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
                                second_policy=second_policy,
                                iteration_limit_ratio=iteration_limit_ratio,
                                num_minimum_children=num_minimum_children)
    current_best_arc = None
    current_best_node = None
    current_best_reward = None
    controller.setRoot()
    pool_size = len(op_pool)
    params = init_params
    for epoch in range(num_iterations):
        start = time.time()
        arcs, nodes = [], []
        if epoch<num_warmup_iterations:
            print("="*10+"Warming at Epoch {}/{}, Total Warmup Epochs: {}, Pool Size: {}".format(epoch+1,
                                                                                                 num_iterations,
                                                                                                 num_warmup_iterations,
                                                                                                 pool_size)+"="*10)
            for _ in range(arc_batchsize):
                k, node = controller.randomSample()
                arcs.append(k)
                nodes.append(node)
        else:
            print("=" * 10 + "Searching at Epoch {}/{}, Pool Size: {}".format(epoch + 1, num_iterations, pool_size) + "=" * 10)
            for _ in range(arc_batchsize):
                k, node = controller.sampleArc(params)
                arcs.append(k)
                nodes.append(node)
        print("Batch Training, {} Iterations in Total...".format(super_circ_train_iterations))
        params, loss_list = batch_training(
            initial_param_proxy=params,
            circ_constructor=model,
            num_epochs=super_circ_train_iterations,
            k_list=arcs,
            op_pool=op_pool,
            training_data=data,
            optimizer_callable=super_circ_train_optimizer,
            lr = super_circ_train_lr,
            circ_noise_factor=super_circ_train_gradient_noise_factor
        )
        reward_list = [1-l for l in loss_list]
        for r, node in zip(reward_list, nodes):
            controller.backPropagate(node, r)
        current_best_arc, current_best_node = controller.exploitArc(params)
        current_best_reward = controller.simulation(current_best_arc, params)
        end = time.time()
        print("Prune Count: {}".format(controller.prune_counter))
        print("Current Best Reward: {}".format(current_best_reward))
        print("Current Best k:\n", current_best_arc)
        print("Current Ops:")
        print(current_best_node.state)
        print("=" * 10 + "Epoch Time: {}".format(end - start) + "=" * 10)

    return params, current_best_arc, current_best_node, current_best_reward

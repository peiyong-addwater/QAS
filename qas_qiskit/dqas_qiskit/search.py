import numpy as np
import jax.numpy as jnp
import jax
import time
import optax
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
    AnyStr
)
from .prob_models import (
    ProbModelBaseClass,
    IndependentCategoricalProbabilisticModel,
    categorical_sample)
from .circuits import (
    QCircFromK,
    BitFlipSearchDensityMatrixNoiseless,
    SIMPLE_DATASET_BIT_FLIP,
    FiveBitCodeSearchDensityMatrixNoiseless,
    SIMPLE_DATASET_FIVE_BIT_CODE
)
from .standard_ops import GatePool, default_complete_graph_parameterized_pool, default_complete_graph_non_parameterized_pool

def train_circuit(num_epochs:int, circ_constructor:Callable, init_params:np.ndarray, k:List[int], op_pool:GatePool,
                  training_data:List[List],opt=optax.adam, lr:float=0.1,
                  early_stopping_threshold:Optional[float]=None,
                  verbose:int=0, early_stopping_avg_num_epochs:int=4):
    p = init_params.shape[0]
    c = init_params.shape[1]
    l = init_params.shape[2]
    if early_stopping_threshold is not None:
        assert num_epochs>=10
        assert early_stopping_avg_num_epochs<num_epochs//2
    assert c == len(op_pool)
    assert p == len(k)
    assert len(training_data) == 2
    params = init_params
    optimizer = opt(lr)
    opt_state = optimizer.init(params)
    if verbose>0:
        print("Starting Circuit Optimization for Max {} Epochs.........".format(num_epochs))
        if verbose>1:
            print("k={}".format(k))
    loss_list = []
    for i in range(num_epochs):
        input_states = training_data[0]
        target_states = training_data[1]
        circ = circ_constructor(p,c,l,k, op_pool)
        loss = circ.get_loss(params, input_states, target_states)
        loss_list.append(loss)
        if verbose>0 and verbose<=1:
            print(
                "Epoch {}, Loss {:.8f}".format(i+1, loss)
            )
        if verbose>1:
            print("==========Epoch {}==========".format(i+1))
            print("Gate Sequence: {}".format(circ.get_circuit_ops(params)))
            print("Loss: {}".format(loss))
        if early_stopping_threshold is not None and i>early_stopping_avg_num_epochs:
            running_avg = np.average(loss_list[-early_stopping_avg_num_epochs:])
            if running_avg <= early_stopping_threshold:
                if verbose>0:
                    print("**"*20)
                    print("Early Stopped at Epoch {}".format(i+1))
                    print("**" * 20)
                break
        grads = circ.get_gradient(params, input_states, target_states)
        grads = jnp.nan_to_num(grads)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        params = np.array(params)

    final_param = params
    final_loss = loss_list[-1]
    final_circ = circ_constructor(p,c,l,k, op_pool)
    final_op_list = final_circ.get_circuit_ops(final_param)
    if verbose>0:
        print("-="*20)
        print("Circuit Training Finished. Final Loss: {}.\n Final Ops:{}".format(final_loss, final_op_list))
        print("-=" * 20)

    return final_param, final_circ, final_op_list, final_loss


#TODO: Early Stopping for Circuit Search
def dqas_qiskit(num_epochs:int,training_data:List[List], init_prob_params:np.ndarray, init_circ_params:np.ndarray,
         op_pool:GatePool, search_circ_constructor:Callable, prob_model=IndependentCategoricalProbabilisticModel,
        circ_lr=0.1, prob_lr = 0.1, circ_opt = optax.adam, prob_opt = optax.adam,
                prob_train_k_num_samples:Optional[int]=None,
         train_circ_in_between_epochs:Optional[int]=None, verbose:int = 0, parameterized_circuit:bool = True):

    p = init_circ_params.shape[0]
    c = init_circ_params.shape[1]
    l = init_circ_params.shape[2]

    prob_params = init_prob_params
    circ_params = init_circ_params

    assert len(training_data) == 2

    if train_circ_in_between_epochs is not None:
        assert train_circ_in_between_epochs > 0
        assert train_circ_in_between_epochs <= 50
    if prob_train_k_num_samples is not None:
        assert prob_train_k_num_samples > 0

    optimizer_for_circ = circ_opt(circ_lr)
    optimizer_for_prob = prob_opt(prob_lr)
    opt_state_circ = optimizer_for_circ.init(circ_params)
    opt_state_prob = optimizer_for_prob.init(prob_params)
    loss_list = []
    if verbose>0:
        print("Starting Circuit Search for Max {} Epochs.........".format(num_epochs))
    for i in range(num_epochs):
        if verbose>1:
            print("===================== Epoch {} =====================".format(i+1))
        epoch_start = time.time()
        pb = prob_model(prob_params)
        # update the parameters for the prob dist first
        if prob_train_k_num_samples is not None:
            sampled_k_list = pb.sample_k(prob_train_k_num_samples)
            prob_losses = [search_circ_constructor(p,c,l,k,op_pool).get_loss(circ_params, training_data[0],
                                                                             training_data[1]) for k in sampled_k_list]
            prob_gradients = [jnp.nan_to_num(pb.get_gradient(prob_losses[i], sampled_k_list[i]))
                              for i in range(prob_train_k_num_samples)]
            prob_gradients = jnp.stack(prob_gradients, axis=0)
            avg_prob_gradients = jnp.mean(prob_gradients, axis=0)
            prob_model_updates, opt_state_prob = optimizer_for_prob.update(avg_prob_gradients, opt_state_prob)
            prob_params = optax.apply_updates(prob_params, prob_model_updates)
        else:
            # choose k according to the probability matrix
            prob_mat = pb.get_prob_matrix()
            chosen_k = jnp.argmax(prob_mat, axis=1)
            chosen_k = [int(c) for c in chosen_k]
            prob_loss = search_circ_constructor(p,c,l,chosen_k,op_pool).get_loss(circ_params, training_data[0],
                                                                             training_data[1])
            prob_gradient = jnp.nan_to_num(pb.get_gradient(prob_loss, chosen_k))
            prob_model_updates, opt_state_prob = optimizer_for_prob.update(prob_gradient, opt_state_prob)
            prob_params = optax.apply_updates(prob_params, prob_model_updates)

        # then update the circuit parameters according to the updated prob parameters
        if train_circ_in_between_epochs is not None and parameterized_circuit:
            if verbose>0:
                print(">>>"*20)
                print("Update the Parameters in the Circuit for {} Iterations...".format(train_circ_in_between_epochs))
            new_pb = prob_model(prob_params)
            new_prob_mat = new_pb.get_prob_matrix()
            best_k = jnp.argmax(new_prob_mat, axis=1)
            best_k = [int(c) for c in best_k]
            circ_params, circ, _, loss = train_circuit(
                train_circ_in_between_epochs, search_circ_constructor, circ_params, best_k, op_pool, training_data,
                verbose=1, lr = 0.05
            )
            loss_list.append(loss)
            if verbose>0:
                print("Between-Iteration Circuit Training Finished! Updated Loss = {:.8f}".format(loss))
                print(">>>" * 20)

        elif train_circ_in_between_epochs is None and parameterized_circuit:
            # only update the parameters of the circuit for one iteration
            new_pb = prob_model(prob_params)
            new_prob_mat = new_pb.get_prob_matrix()
            best_k = jnp.argmax(new_prob_mat, axis=1)
            best_k = [int(c) for c in best_k]
            circ = search_circ_constructor(p,c,l,best_k,op_pool)
            loss = circ.get_loss(circ_params, training_data[0], training_data[1])
            loss_list.append(loss)
            circ_gradient = circ.get_gradient(circ_params, training_data[0], training_data[1])
            circ_gradient = jnp.nan_to_num(circ_gradient)
            circ_updates, opt_state_circ = optimizer_for_circ.update(circ_gradient, opt_state_circ)
            circ_params = optax.apply_updates(circ_params, circ_updates)
            circ_params = np.array(circ_params)
        else:
            # no parameters in the gates
            new_pb = prob_model(prob_params)
            new_prob_mat = new_pb.get_prob_matrix()
            best_k = jnp.argmax(new_prob_mat, axis=1)
            best_k = [int(c) for c in best_k]
            circ = search_circ_constructor(p, c, l, best_k, op_pool)
            loss = circ.get_loss(circ_params, training_data[0], training_data[1])
            loss_list.append(loss)
        epoch_end = time.time()
        if verbose>0 and verbose<=1:
            print(
                "Epoch {}, Loss {:.6f}, Epoch Time: {}".format(i+1, loss, epoch_end-epoch_start)
            )
        if verbose>1:
            print("k={}".format(best_k))
            print("Gate Sequence: {}".format(circ.get_circuit_ops(circ_params)))
            print("Loss: {:.8f}".format(loss))
            print("Epoch Time: {:.4f} seconds".format(epoch_end-epoch_start))

    final_circ_param = circ_params
    final_prob_param = prob_params
    final_loss = loss_list[-1]
    final_prob_model = prob_model(final_prob_param)
    final_prob_mat = final_prob_model.get_prob_matrix()
    final_k = jnp.argmax(final_prob_mat, axis=1)
    final_k = [int(c) for c in final_k]
    final_circ = search_circ_constructor(p, c, l, final_k, op_pool)
    final_op_list = final_circ.get_circuit_ops(final_circ_param)
    if verbose>0:
        print("-="*20)
        print("Circuit Search Finished.\nFinal Loss: {}.\nFinal Ops:\n{}".format(final_loss, final_op_list))
        print("Final k:\n{}".format(final_k))
        print("Final Probability Matrix\n{}".format(final_prob_mat))
        print("Final Prob Model Parameter\n{}".format(final_prob_param))
        print("-=" * 20)

    return final_prob_param, final_circ_param, final_prob_model, final_circ, final_k, final_op_list, final_loss

"""
pool =default_complete_graph_parameterized_pool(5)
p = 30
c = len(pool)
l = 3
param = np.random.randn(p*c*l).reshape((p,c,l))
a = np.zeros(p*c)
a = a.reshape((p,c))
final_prob_param, final_circ_param, final_prob_model, final_circ, final_k, final_op_list, final_loss= dqas_qiskit(
    50, SIMPLE_DATASET_FIVE_BIT_CODE, a, param, pool, FiveBitCodeSearchDensityMatrixNoiseless,
    IndependentCategoricalProbabilisticModel,
    prob_train_k_num_samples=200, verbose=2,train_circ_in_between_epochs=10,parameterized_circuit=True
)

"""




"""
pool = default_complete_graph_parameterized_pool(3)
p = 7
c = len(pool)
l = 3
print(pool)
k = [3,4,4,6,8,3,2]
p = np.random.rand(p*c*l).reshape((p,c,l))
final_param, final_circ, final_op_list, final_loss = train_circuit(50, BitFlipSearchDensityMatrix,p,k,pool,
                                                                   SIMPLE_DATASET_BIT_FLIP, verbose=2,
                                                                   early_stopping_threshold=0.01)
"""
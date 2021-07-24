import numpy as np
import jax.numpy as jnp
import jax
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
from qiskit.quantum_info import DensityMatrix


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
        epoch_start = time.time()
        input_states = training_data[0]
        target_states = training_data[1]
        circ = circ_constructor(p,c,l,k, op_pool)
        loss = circ.get_loss(params, input_states, target_states)
        loss_list.append(loss)
        grads = circ.get_gradient(params, input_states, target_states)
        grads = jnp.nan_to_num(grads)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        params = np.array(params)
        epoch_end = time.time()
        if verbose>0 and verbose<=1:
            print(
                "Epoch {}, Loss {:.8f}, Time{}".format(i+1, loss, epoch_end-epoch_start)
            )
        if verbose>1:
            print("==========Epoch {} Time {}==========".format(i+1, epoch_end-epoch_start))
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

    final_param = params
    final_loss = loss_list[-1]
    final_circ = circ_constructor(p,c,l,k, op_pool)
    final_op_list = final_circ.get_circuit_ops(final_param)
    if verbose>0:
        print("-="*20)
        print("Circuit Training Finished. Final Loss: {}.\n Final Ops:{}".format(final_loss, final_op_list))
        print("-=" * 20)

    return final_param, final_circ, final_op_list, final_loss,loss_list

def _entropy_of_prob_mat_categorical(prob_mat:Union[np.ndarray, jnp.ndarray]):
    p = prob_mat.shape[0]
    c = prob_mat.shape[1]
    p_list = []
    for i in range(p):
        c_list = [i for i in range(c)]
        p_list.append(c_list)
    combinations = [elem for elem in itertools.product(*p_list)]

    def single_prob(combo):
        prob = 1
        for i in range(p):
            prob = prob * prob_mat[i, combo[i]]
        return prob*np.log(1/prob)

    probs = Parallel(n_jobs=-1, verbose=0)(delayed(single_prob)(combo) for combo in combinations)
    hp = np.sum(probs)
    return np.float(hp)


def _circ_obj_get_loss_dm(circ_obj, circ_params, init_state, target_state:List[DensityMatrix]):
    return circ_obj.get_loss(circ_params, init_state, target_state)

def _circ_obj_get_gradient_dm(circ_obj, circ_params, init_state, target_state:List[DensityMatrix]):
    return circ_obj.get_gradient(circ_params, init_state, target_state)


def dqas_qiskit(num_epochs:int,
                   training_data:List[List],
                   init_prob_params:np.ndarray,
                   init_circ_params:np.ndarray,
                   op_pool:GatePool,
                   search_circ_constructor:Callable,
                   prob_model=IndependentCategoricalProbabilisticModel,
                   circ_lr=0.1,
                   prob_lr = 0.1,
                   circ_opt = optax.adam,
                   prob_opt = optax.adam,
                   batch_k_num_samples:int=200,
                   verbose:int = 0,
                   parameterized_circuit:bool = True,
                   prob_grad_noise_factor = 1/50):

    p = init_circ_params.shape[0]
    c = init_circ_params.shape[1]
    l = init_circ_params.shape[2]

    prob_params = init_prob_params
    circ_params = init_circ_params

    assert len(training_data) == 2
    assert batch_k_num_samples > 0

    optimizer_for_circ = circ_opt(circ_lr)
    optimizer_for_prob = prob_opt(prob_lr)
    opt_state_circ = optimizer_for_circ.init(circ_params)
    opt_state_prob = optimizer_for_prob.init(prob_params)
    loss_list = []
    sample_batch_avg_loss = 0
    loss_std = []
    prob_params_list = []
    optimal_circuit_loss = []
    pb = prob_model(prob_params)
    if verbose>0:
        print("Starting Circuit Search for Max {} Epochs.........".format(num_epochs))
    for i in range(num_epochs):
        if verbose>1:
            print("=============================== Epoch {}/{} ===============================".format(i+1, num_epochs))
        epoch_start = time.time()
        # update the circuit parameters and the prob parameters
        # sample a batch of k
        if verbose > 0:
            print("Sample Circuits for Batch Size: {}".format(batch_k_num_samples))
        sampled_k_list = pb.sample_k(batch_k_num_samples)
        #sampled_k_prob = Parallel(n_jobs=-1, verbose=0)(delayed(pb.get_prob_for_k)(k) for k in sampled_k_list)

        sampled_circs = [search_circ_constructor(p, c, l, k, op_pool) for k in sampled_k_list]


        if verbose > 0:
            print("Calculating losses...")
        batch_losses = Parallel(n_jobs=-1, verbose=0)(delayed(_circ_obj_get_loss_dm)(constructed_circ, circ_params,
                                            training_data[0], training_data[1]) for constructed_circ in sampled_circs)
        batch_losses = np.nan_to_num(batch_losses, nan=1.0) # deal with NaN in losses
        batch_loss_std = jnp.std(batch_losses)
        loss_std.append(float(batch_loss_std))
        if verbose > 0:
            print("Standard deviation of the losses in current batch: {}".format(batch_loss_std))

        if verbose > 0:
            print("Loss Calculation Finished!")

        prob_losses_modified = [batchloss - sample_batch_avg_loss for batchloss in batch_losses]
        sample_batch_avg_loss = np.average(batch_losses)
        if verbose > 0:
            print("Calculating gradients for prob model parameters...")
        prob_gradients = Parallel(n_jobs=-1, verbose=0)(delayed(pb.get_gradient)(prob_losses_modified[i],
                                                                                 sampled_k_list[i]) for i in
                                                        range(batch_k_num_samples))
        prob_gradients = Parallel(n_jobs=-1, verbose=0)(delayed(jnp.nan_to_num)(c) for c in prob_gradients)

        prob_gradients = jnp.stack(prob_gradients, axis=0)
        prob_gradients = jnp.mean(prob_gradients, axis=0)
        # add noise to probability parameter gradients
        seed = np.random.randint(0, 10000000000000)
        key = jax.random.PRNGKey(seed)
        noise = jax.random.normal(key=key, shape=(p, c))
        #print(noise/50)
        prob_gradients = prob_gradients+noise*prob_grad_noise_factor

        if verbose > 0:
            print("Prob Model Param Gradients Calculation Finished!")

        if parameterized_circuit:
            if verbose > 0:
                print("Calculating gradients for circuit parameters...")
            circ_batch_gradients = Parallel(n_jobs=1, verbose=0)(delayed(_circ_obj_get_gradient_dm)(constructed_circ,
                        circ_params, training_data[0], training_data[1]) for constructed_circ in sampled_circs)

            if verbose > 0:
                print("Circuit Param Gradients Calculation Finished!")

        else:
            circ_batch_gradients = [np.zeros(p,c,l) for _ in sampled_k_list]


        circ_batch_gradients = jnp.stack(circ_batch_gradients, axis=0)
        circ_batch_gradients = jnp.nan_to_num(circ_batch_gradients)
        circ_gradient = jnp.mean(circ_batch_gradients, axis=0)

        # update the parameters
        prob_model_updates, opt_state_prob = optimizer_for_prob.update(prob_gradients, opt_state_prob)
        prob_params = optax.apply_updates(prob_params, prob_model_updates)

        circ_updates, opt_state_circ = optimizer_for_circ.update(circ_gradient, opt_state_circ)
        circ_params = optax.apply_updates(circ_params, circ_updates)
        # add some noise to the circuit parameter
        # seed = np.random.randint(0, 100)
        # key = jax.random.PRNGKey(seed)
        # noise = jax.random.normal(key=key, shape=(p, c, l))
        # circ_params = circ_params+noise/10
        circ_params = np.array(circ_params)

        loss_list.append(sample_batch_avg_loss)

        pb = prob_model(prob_params)
        new_prob_mat = pb.get_prob_matrix()

        if verbose>=10:
            print("Prob Param Gradient:"),
            print(np.array(prob_gradients))
            print("Prob Matrix:")
            print(np.array(new_prob_mat))

        prob_params_list.append(np.array(prob_params))
        best_k = jnp.argmax(new_prob_mat, axis=1)
        best_k = [int(c) for c in best_k]
        best_circ = search_circ_constructor(p, c, l, best_k, op_pool)
        optimal_circuit_loss.append(float(best_circ.get_loss(circ_params,
                                                                training_data[0], training_data[1])))
        epoch_end = time.time()


        if verbose>1:
            print("New Optimal k={}".format(best_k))
            print("New Optimal Gate Sequence: {}".format(best_circ.get_circuit_ops(circ_params)))
            print(
                ">>>>>>Loss On New Optimal Gate Sequence: {:.8f}".format(optimal_circuit_loss[-1])
            )
            print(">>>>>>>>Batch Avg Loss on Samples: {:.6f}<<<<<<<<".format(np.average(batch_losses)))
            epoch_end = time.time()
            print(
                "Epoch {}, Time: {}".format(i + 1, epoch_end - epoch_start)
            )
        if verbose>0 and verbose<=1:
            print(
                "Epoch {}, Batch Avg Loss on Samples: {:.6f}, Epoch Time: {}".format(i+1, loss_list[-1],
                                                                                     epoch_end-epoch_start)
            )

    final_circ_param = np.array(circ_params)
    final_prob_param = np.array(prob_params)
    final_loss = loss_list[-1]
    final_prob_model = prob_model(final_prob_param)
    final_prob_mat = np.array(final_prob_model.get_prob_matrix())
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

    return final_prob_param, final_circ_param, final_prob_model, final_circ, final_k, final_op_list, final_loss, \
           loss_list, loss_std, prob_params_list


















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
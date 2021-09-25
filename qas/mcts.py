from copy import deepcopy
import random
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
import numpy as np
import jax
import jax.numpy as jnp
import pennylane as qml
import pennylane.numpy as pnp
from abc import ABC, abstractmethod
from .qml_ops import QMLGate, QMLPool, SUPPORTED_OPS_DICT
from .models import ModelFromK
import optax
import os
import time
from joblib import Parallel, delayed


class StateOfMCTS(ABC):

    @abstractmethod
    def getLegalActions(self, *args):
        raise NotImplementedError
    @abstractmethod
    def takeAction(self,*args):
        raise NotImplementedError
    @abstractmethod
    def isTerminal(self):
        raise NotImplementedError
    @abstractmethod
    def getReward(self):
        raise NotImplementedError
    @abstractmethod
    def __repr__(self):
        raise NotImplementedError
    @abstractmethod
    def getCurrK(self):
        raise NotImplementedError


class QMLState(StateOfMCTS):
    def __init__(self, current_k:List[int]=[], op_pool:QMLPool=None, maxDepth = 30, qubit_with_actions:Set = None):
        self.max_depth = maxDepth
        self.current_k = current_k
        self.current_depth = len(self.current_k)
        assert self.current_depth <= self.max_depth
        self.pool_obj = op_pool
        self.op_name_dict = op_pool.pool
        self.pool_keys = list(op_pool.pool.keys())
        self.state = None
        self.qubit_with_actions = qubit_with_actions if qubit_with_actions is not None else set()

    def getLegalActions(self):
        if self.current_depth == self.max_depth:
            return []
        actions = []
        for key in self.pool_keys:
            if self.verifyDesirableAction(key):
                actions.append(key)
        return actions

    def verifyDesirableAction(self, action:int):
        # many change with different tasks
        op = self.op_name_dict[action]
        assert len(op.keys()) == 1 # in case some weird things happen
        op_name = list(op.keys())[0]
        op_qubit = op[op_name]
        op_obj = SUPPORTED_OPS_DICT[op_name]
        op_num_params = op_obj.num_params
        # don't want two consecutive parameterized gates or Pauli gates or CNOT gates
        if len(self.current_k)>=1 and action==self.current_k[-1]:
            if op_num_params>0:
                return False
            elif op_name in ['PauliX', 'PauliY', 'PauliZ', 'CNOT', 'CZ', 'CY', 'Hadamard']:
                return False
        if len(op_qubit) == 2 and op_qubit[0] not in self.qubit_with_actions:
            return False
        return True

    def takeAction(self, action:int):
        new_qubit_with_actions = deepcopy(self.qubit_with_actions)
        op = self.op_name_dict[action]
        assert len(op.keys()) == 1  # in case some weird things happen
        op_name = list(op.keys())[0]
        op_qubit = op[op_name]
        new_qubit_with_actions.add(op_qubit[-1])
        new_path = self.current_k + [action]
        new_state = QMLState(current_k=new_path, op_pool=self.pool_obj, maxDepth=self.max_depth,
                             qubit_with_actions=new_qubit_with_actions)
        new_state.state = action
        return new_state

    def isTerminal(self):
        return self.current_depth>=self.max_depth

    def getReward(self):
        return 0.

    def getCurrK(self):
        return self.current_k

    def __repr__(self):
        disp = "LegalControlQubits: {}\nLegalActions: {}\n".format(self.qubit_with_actions,self.getLegalActions())
        for i, p in enumerate(self.current_k):
            disp = disp + "OpAtDepth: {}\tOpKey: {}\tOpName: {}\n".format(i, p, self.op_name_dict[p])
        return disp

class TreeNode():
    # https://github.com/pbsinclair42/MCTS/blob/master/mcts.py
    def __init__(self, state:StateOfMCTS, parent):
        self.state = state
        self.isTerminal = state.isTerminal()
        self.isFullyExpanded = self.isTerminal
        self.parent = parent
        self.numVisits = 0
        self.totalReward = 0
        self.children = {}

    def __repr__(self):
        return "State: {}\nParent: {}\nNum Visits: {}\nTotal Reward: {}\nChildern: {}\nisTerminal: {}\nisFullyExpanded: {}".format(self.state, self.parent, self.numVisits, self.totalReward, self.children, self.isTerminal, self.isFullyExpanded)

class MCTSController():
    # https://github.com/pbsinclair42/MCTS/blob/master/mcts.py
    def __init__(self,
                 model,
                 op_pool,
                 target_circuit_depth:int,
                 reward_penalty_function:Optional[Callable]=None, # should take simulation reward and node as input and return new reward
                 initial_legal_control_qubit_choice:Optional=None,
                 alpha = 1/np.sqrt(2),
                 prune_reward_ratio=0.9,
                 max_visits_prune_threshold=50,
                 min_num_children=5,
                 sampling_execute_rounds=20,
                 exploit_execute_rounds=200,
                 sample_policy='local_optimal',
                 exploit_policy='local_optimal'
                 ):
        self.model = model
        self.pool = op_pool
        self.pool_dict = op_pool.pool
        self.pool_keys = list(self.pool_dict.keys())
        self.max_depth = target_circuit_depth
        self.penalty_func = reward_penalty_function
        self.initial_legal_control_qubit_choice = set() if initial_legal_control_qubit_choice is None else initial_legal_control_qubit_choice
        self.alpha = alpha
        self.first_policy = sample_policy
        self.second_policy = exploit_policy
        self.prune_reward_ratio = prune_reward_ratio
        self.max_visits_prune_threshold = max_visits_prune_threshold
        self.min_num_children = min_num_children
        self.sampling_execute_rounds = sampling_execute_rounds
        self.exploit_execute_rounds=exploit_execute_rounds
        self.prune_counter = 0
        self.initial_state = QMLState(op_pool=self.pool, maxDepth=self.max_depth, qubit_with_actions=self.initial_legal_control_qubit_choice)

    def _reset(self):
        self.root = TreeNode(self.initial_state, None)
        self.prune_counter = 0

    def setRoot(self):
        self.root = TreeNode(self.initial_state, None)

    def getActionFromBestChild(self, root:TreeNode, best_child):
        for action, node in root.children.items():
            if node is best_child:
                return action

    def expand(self, node:TreeNode):
        actions = node.state.getLegalActions()
        random.shuffle(actions)
        for action in actions:
            if action not in node.children:
                new_node = TreeNode(node.state.takeAction(action), node)
                node.children[action] = new_node
                if len(actions) == len(node.children):
                    node.isFullyExpanded = True
                return new_node

    def backPropagate(self, node:TreeNode, reward):
        while node is not None:
            node.numVisits = node.numVisits + 1
            node.totalReward = node.totalReward+ reward
            node = node.parent

    def simulationWithSuperCircuitParamsAndK(self, k:List[int], params):
        assert len(k) == self.max_depth
        p, c, l = params.shape[0], params.shape[1], params.shape[2]
        circ = self.model(p, c, l, k, self.pool)
        r = circ.getReward(params)
        return r

    def randomSample(self):
        node = self.root
        while not node.state.isTerminal():
            if not node.isFullyExpanded:
                node = self.expand(node)
            else:
                action, node = random.choice(list(node.children.items()))
        return node.state.getCurrK(), node

    def getBestChild(self, node:TreeNode, alpha, policy='local_optimal'):
        if len(list(node.children.values())) == 0:
            print(node)
            raise ValueError("No Legal Child for Node at: %s"%node.state.getCurrK())
        if policy == 'random':
            return random.choice(list(node.children.values()))
        best_value = float("-inf")
        best_nodes, suboptimal_nodes = [], []
        for child in node.children.values():
            node_value = child.totalReward / child.numVisits + alpha * np.sqrt(
                2 * np.log(node.numVisits) / child.numVisits)
            if node_value > best_value:
                if best_value == float("-inf"):
                    suboptimal_nodes = [child]
                else:
                    suboptimal_nodes = best_nodes
                best_value = node_value
                best_nodes = [child]
            elif node_value == best_value:
                best_nodes.append(child)
        if policy == 'local_sub_optimal':
            node = np.random.choice(suboptimal_nodes)
        elif policy == 'local_optimal':
            node = np.random.choice(best_nodes)
        else:
            raise ValueError("No such policy: %s" % policy)
        return node

    def selectNode(self, node:TreeNode):
        if node.isFullyExpanded:
            # pruning
            node_avg_reward = node.totalReward/node.numVisits
            threshold = node_avg_reward*self.prune_reward_ratio
            pruned_key = []
            for key in list(node.children.keys()):
                child = node.children[key]
                child_avg_reward = child.totalReward/child.numVisits
                if child_avg_reward<threshold and child.numVisits>self.max_visits_prune_threshold:
                    pruned_key.append(key)
            random.shuffle(pruned_key)
            for i, item in enumerate(pruned_key):
                if i > len(pruned_key)-self.min_num_children:
                    break
                node.children.pop(item)
                self.prune_counter+=1
            node = self.getBestChild(node, self.alpha, policy=self.first_policy)
        else:
            node = self.expand(node)
        return node

    def executeRoundWithSuperCircParamsFromAnyNode(self,node:TreeNode, params):
        if node is None:
            node = self.root
        while not node.state.isTerminal():
            node = self.selectNode(node)
        reward = self.simulationWithSuperCircuitParamsAndK(node.state.getCurrK(), params)
        reward = reward if self.penalty_func is None else self.penalty_func(reward, node)
        self.backPropagate(node, reward)
        return node

    def sampleArcWithSuperCircParams(self, params):
        curr = self.root
        for i in range(self.sampling_execute_rounds):
            #execute multiple rounds, then sample the tree based on updated rewards
            self.executeRoundWithSuperCircParamsFromAnyNode(node=curr, params=params)
        while not curr.state.isTerminal():
            curr = self.getBestChild(curr, self.alpha, policy=self.first_policy)
        return curr.state.getCurrK(), curr

    def exploitArcWithSuperCircParams(self, params):
        curr = self.root
        while not curr.state.isTerminal():
            for i in range(self.exploit_execute_rounds):
                self.executeRoundWithSuperCircParamsFromAnyNode(node=curr, params=params)
            curr = self.getBestChild(curr, 0, policy=self.second_policy) # alpha = 0, no exploration
        return curr.state.getCurrK(), curr

def getGradientFromModel(model, params):
    return model.getGradient(params)

def getLossFromModel(model, params):
    return model.getLoss(params)

def getRewardFromModel(model, params):
    return model.getReward(params)

def search(
        model,
        op_pool,
        target_circuit_depth,
        init_qubit_with_controls:Set,
        init_params:Union[np.ndarray, jnp.ndarray, Sequence],
        num_iterations = 500,
        num_warmup_iterations = 20,
        super_circ_train_optimizer = optax.adam,
        super_circ_train_gradient_noise_factor = 1/100,
        super_circ_train_lr = 0.01,
        penalty_function:Callable=None,
        arc_batchsize = 200,
        alpha_max = 2,
        alpha_min=1 / np.sqrt(2),
        prune_constant_max=0.9,
        prune_constant_min = 0.5,
        max_visits_prune_threshold=100,
        min_num_children=5,
        sampling_execute_rounds=200,
        exploit_execute_rounds=200,
        sample_policy='local_optimal',
        exploit_policy='local_optimal'
):
    p = target_circuit_depth
    l = init_params.shape[2]
    c = len(op_pool)
    assert p == init_params.shape[0]
    assert c == init_params.shape[1]
    assert alpha_min <= alpha_max
    assert prune_constant_min <= prune_constant_max
    controller = MCTSController(
        model=model,
        op_pool=op_pool,
        target_circuit_depth=target_circuit_depth,
        reward_penalty_function=penalty_function,
        initial_legal_control_qubit_choice=init_qubit_with_controls,
        alpha=alpha_max,
        prune_reward_ratio=prune_constant_min,
        max_visits_prune_threshold=max_visits_prune_threshold,
        min_num_children=min_num_children,
        sampling_execute_rounds=sampling_execute_rounds,
        exploit_execute_rounds=exploit_execute_rounds,
        sample_policy=sample_policy,
        exploit_policy=exploit_policy
    )
    current_best_arc = None
    current_best_node = None
    current_best_reward = None
    controller.setRoot()
    pool_size = len(op_pool)
    params = init_params
    optimizer = super_circ_train_optimizer(super_circ_train_lr)
    opt_state = optimizer.init(params)
    best_rewards = []
    for epoch in range(num_iterations):
        start = time.time()
        arcs, nodes = [], []
        if epoch<num_warmup_iterations:
            print("="*10+"Random Sampling at Epoch {}/{}, Total Warmup Epochs: {}, Pool Size: {}, "
                         "Arc Batch Size: {}, Sampling Rounds: {}, Exploiting Rounds: {}"
                  .format(epoch+1,
                          num_iterations,
                          num_warmup_iterations,
                          pool_size,
                          arc_batchsize,
                          sampling_execute_rounds,
                          exploit_execute_rounds)
                  +"="*10)
            for _ in range(arc_batchsize):
                k, node = controller.randomSample()
                arcs.append(k)
                nodes.append(node)
        else:
            print("=" * 10 + "Searching (Sampling According to UCT and CMAB) at Epoch {}/{}, Pool Size: {}, "
                             "Arc Batch Size: {}, Sampling Rounds: {}, Exploiting Rounds: {}"
                  .format(epoch + 1,
                          num_iterations,
                          pool_size,
                          arc_batchsize,
                          sampling_execute_rounds,
                          exploit_execute_rounds)
                  + "=" * 10)
            controller._reset() # CMAB reset
            new_alpha = alpha_max - (alpha_max - alpha_min) / (num_iterations - num_warmup_iterations) * (
                        epoch + 1 - num_warmup_iterations)
            controller.alpha = new_alpha  # alpha decreases as epoch increases
            new_prune_rate = prune_constant_min + (prune_constant_max - prune_constant_min) / (
                        num_iterations - num_warmup_iterations) * (epoch + 1 - num_warmup_iterations)
            controller.prune_reward_ratio = new_prune_rate  # prune rate increases as epoch increases
            for _ in range(arc_batchsize):
                k, node = controller.sampleArcWithSuperCircParams(params)
                arcs.append(k)
                nodes.append(node)
        print("Batch Training, Size = {}, Update the Parameter Pool for One Iteration".format(arc_batchsize))
        batch_models = [model(p,c,l,k,op_pool) for k in arcs]
        #batch_gradients = Parallel(n_jobs=-1, verbose=0)(
        #    delayed(getGradientFromModel)(constructed_model, params) for constructed_model in batch_models
        #)
        batch_gradients = [getGradientFromModel(constructed_model, params) for constructed_model in batch_models]
        batch_gradients = jnp.stack(batch_gradients, axis=0)
        batch_gradients = jnp.nan_to_num(batch_gradients)
        batch_gradients = jnp.mean(batch_gradients, axis=0)
        # add some noise to the circuit gradient
        seed = np.random.randint(0, 1000000000)
        key = jax.random.PRNGKey(seed)
        noise = jax.random.normal(key, shape=(p, c, l))
        batch_gradients = batch_gradients + noise*super_circ_train_gradient_noise_factor
        circ_updates, opt_state = optimizer.update(batch_gradients, opt_state)
        params = optax.apply_updates(params, circ_updates)
        params = np.array(params)
        print("Parameters Updated!")
        print("Calculating Rewards for Sampled Arcs...")
        #reward_list = Parallel(n_jobs=-1, verbose=0)(
        #    delayed(controller.simulationWithSuperCircuitParamsAndK)(k, params) for k in arcs
        #)
        reward_list = [controller.simulationWithSuperCircuitParamsAndK(k, params) for k in arcs]
        for r, node in zip(reward_list, nodes):
            r = penalty_function(r, node) if penalty_function is not None else r
            controller.backPropagate(node, r)
        print("Reward BPed!")
        print("Exploiting and finding the best arc...")
        current_best_arc, current_best_node = controller.exploitArcWithSuperCircParams(params)
        current_best_reward = controller.simulationWithSuperCircuitParamsAndK(current_best_arc, params)
        end = time.time()
        print("Prune Count: {}".format(controller.prune_counter))
        print("Current Best Reward: {}".format(current_best_reward))
        print("Current Best k:\n", current_best_arc)
        print("Current Ops:")
        print(current_best_node.state)
        print("=" * 10 + "Epoch Time: {}".format(end - start) + "=" * 10)
        best_rewards.append((current_best_arc, current_best_reward))

    return params, current_best_arc, current_best_node, current_best_reward, controller, best_rewards

def circuitModelTuning(
        initial_params,
        model,
        num_epochs,
        k,
        op_pool,
        opt_callable=optax.adam,
        lr = 0.01,
        grad_noise_factor = 1/100,
        verbose = 1
):
    p, c, l = initial_params.shape[0], initial_params.shape[1], initial_params.shape[2]
    circ = model(p, c, l, k, op_pool)
    optimizer = opt_callable(lr)
    circ_params = initial_params
    opt_state = optimizer.init(circ_params)
    loss_list = []
    for epoch in range(num_epochs):
        loss = circ.getLoss(circ_params)
        loss_list.append(loss)
        if verbose >= 1:
            print('Training Circuit at Epoch {}/{}; Loss: {}'.format(epoch + 1, num_epochs, loss))
        gradients = circ.getGradient(circ_params)
        gradients = jnp.nan_to_num(gradients)
        seed = np.random.randint(0, 1000000000)
        key = jax.random.PRNGKey(seed)
        noise = jax.random.normal(key, shape=(p, c, l))
        gradients = gradients + noise * grad_noise_factor
        circ_updates, opt_state = optimizer.update(gradients, opt_state)
        circ_params = optax.apply_updates(circ_params, circ_updates)
        circ_params = np.array(circ_params)
    return circ_params, loss_list
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
import pennylane as qml
import pennylane.numpy as pnp
from abc import ABC, abstractmethod
from .qml_ops import QMLGate, QMLPool, SUPPORTED_OPS_DICT
import time
#from memory_profiler import profile
from joblib import Parallel, delayed
from tqdm import tqdm, trange

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
    def __init__(self, current_k:List[int]=[], op_pool:QMLPool=None, maxDepth = 30, qubit_with_actions:Set = None, gate_limit_dict:Optional[dict] = None):
        self.max_depth = maxDepth
        self.current_k = current_k
        self.current_depth = len(self.current_k)
        assert self.current_depth <= self.max_depth
        self.pool_obj = op_pool
        self.op_name_dict = op_pool.pool
        self.pool_keys = list(op_pool.pool.keys())
        self.state = None
        self.qubit_with_actions = qubit_with_actions if qubit_with_actions is not None else set()
        self.gate_limit_dict = gate_limit_dict if gate_limit_dict is not None else {}
        self.gate_count = {}
        for key in self.gate_limit_dict.keys():
            self.gate_count[key] = 0
        for action in current_k:
            op = self.op_name_dict[action]
            assert len(op.keys()) == 1  # in case some weird things happen
            op_name = list(op.keys())[0]
            if op_name in set(self.gate_limit_dict.keys()):
                self.gate_count[op_name] = self.gate_count[op_name] + 1

    def getLegalActions(self):
        if self.current_depth == self.max_depth:
            return []
        actions = []
        for key in self.pool_keys:
            if self.verifyDesirableAction(key):
                actions.append(key)
        return actions

    def stackOpsOnQubit(self, k):
        stacked_ops = [[] for _ in range(self.pool_obj.num_qubits)]
        for action in k:
            op = self.op_name_dict[action]
            assert len(op.keys()) == 1 # in case some weird things happen
            op_name = list(op.keys())[0]
            op_qubit = op[op_name]
            for qubit in op_qubit:
                if op_name != "PlaceHolder":
                    stacked_ops[qubit].append(action)
        return stacked_ops

    def verifyDesirableAction(self, action:int):
        # many change with different tasks
        op = self.op_name_dict[action]
        assert len(op.keys()) == 1 # in case some weird things happen
        op_name = list(op.keys())[0]
        op_qubit = op[op_name]
        op_obj = SUPPORTED_OPS_DICT[op_name]
        op_num_params = op_obj.num_params
        stacked_ops = self.stackOpsOnQubit(self.current_k)
        # don't want two consecutive parameterized gates or Pauli gates or CNOT gates
        if len(self.current_k)>=1:
            if len(op_qubit) == 1:
                if op_num_params>0:
                    if len(stacked_ops[op_qubit[0]])>=1:
                        if stacked_ops[op_qubit[0]][-1] == action:
                            return False
                elif op_name in ['PauliX', 'PauliY', 'PauliZ','Hadamard']:
                    if len(stacked_ops[op_qubit[0]])>=1:
                        if stacked_ops[op_qubit[0]][-1] == action:
                            return False
            if len(op_qubit) == 2:
                if len(stacked_ops[op_qubit[0]])>=1 and len(stacked_ops[op_qubit[1]])>=1:
                    if stacked_ops[op_qubit[0]][-1] == action and stacked_ops[op_qubit[1]][-1] == action:
                        if op_num_params>0: return False
                        if op_name in ['CNOT', 'CZ', 'CY']: return False

        if len(op_qubit) == 2 and op_qubit[0] not in self.qubit_with_actions:
            return False

        # control the number of gates
        if op_name in set(self.gate_limit_dict.keys()):
            if self.gate_count[op_name] >= self.gate_limit_dict[op_name]:
                return False

        return True

    def takeAction(self, action:int):
        new_qubit_with_actions = deepcopy(self.qubit_with_actions)
        op = self.op_name_dict[action]
        assert len(op.keys()) == 1  # in case some weird things happen
        op_name = list(op.keys())[0]
        op_qubit = op[op_name]
        if op_name != "PlaceHolder":
            new_qubit_with_actions.add(op_qubit[-1])
        new_path = self.current_k + [action]
        new_state = QMLState(current_k=new_path, op_pool=self.pool_obj, maxDepth=self.max_depth,
                             qubit_with_actions=new_qubit_with_actions, gate_limit_dict=self.gate_limit_dict)
        new_state.state = action
        return new_state

    def isTerminal(self):
        return self.current_depth>=self.max_depth

    def getReward(self):
        return 0.

    def getCurrK(self):
        return self.current_k

    def __repr__(self):
        disp = ""
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
                 initial_legal_control_qubit_choice:Optional[Set]=None,
                 alpha = 1/np.sqrt(2),
                 prune_reward_ratio=0.9,
                 max_visits_prune_threshold=50,
                 min_num_children=5,
                 sampling_execute_rounds=20,
                 exploit_execute_rounds=200,
                 sample_policy='local_optimal',
                 exploit_policy='local_optimal',
                 gate_limit_dict:Optional[dict] = None
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
        self.initial_state = QMLState(op_pool=self.pool, maxDepth=self.max_depth, qubit_with_actions=self.initial_legal_control_qubit_choice, gate_limit_dict=gate_limit_dict)

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
        self.r = circ.getReward(params)
        return self.r

    def randomSample(self):
        node = self.root
        while not node.state.isTerminal():
            if not node.isFullyExpanded:
                node = self.expand(node)
            else:
                action, node = random.choice(list(node.children.items()))
        return node.state.getCurrK(), node

    def uctSample(self, policy='local_optimal'):
        node = self.root
        while not node.state.isTerminal():
            if not node.isFullyExpanded:
                node = self.expand(node)
            else:
                node = self.getBestChild(node, self.alpha, policy=policy)
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

def getSimulationReward(controller:MCTSController, k:List[int], node:TreeNode, params):
    r = controller.simulationWithSuperCircuitParamsAndK(k, params)
    return (r, k, node)


def search(
        model=None,
        op_pool=None,
        target_circuit_depth:int=None,
        init_qubit_with_controls:Set=None,
        init_params:Union[np.ndarray, pnp.ndarray, Sequence]=None,
        num_iterations = 500,
        num_warmup_iterations = 20,
        super_circ_train_optimizer = qml.AdamOptimizer,
        super_circ_train_gradient_noise_factor = 1/100,
        super_circ_train_lr = 0.01,
        penalty_function:Callable=None,
        gate_limit_dict:Optional[dict] = None,
        warmup_arc_batchsize = 200,
        search_arc_batchsize = 20,
        alpha_max = 2,
        alpha_min=1 / np.sqrt(2),
        prune_constant_max=0.9,
        prune_constant_min = 0.2,
        max_visits_prune_threshold=100,
        min_num_children=5,
        sampling_execute_rounds=200,
        exploit_execute_rounds=200,
        cmab_sample_policy='local_optimal',
        cmab_exploit_policy='local_optimal',
        uct_sample_policy = 'local_optimal',
        verbose = 1
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
        sample_policy=cmab_sample_policy,
        exploit_policy=cmab_exploit_policy,
        gate_limit_dict=gate_limit_dict
    )
    current_best_arc = None
    current_best_node = None
    current_best_reward = None
    controller.setRoot()
    pool_size = len(op_pool)
    params = init_params
    optimizer = super_circ_train_optimizer(super_circ_train_lr)
    best_rewards = []
    for epoch in range(num_iterations):
        start = time.time()
        arcs, nodes = [], []
        if epoch<num_warmup_iterations:
            print("="*10+"Model: {}, Warming Up at Epoch {}/{}, Total Warmup Epochs: {}, Pool Size: {}, "
                         "Arc Batch Size: {}, Exploiting Rounds: {}"
                  .format(model.name,
                        epoch+1,
                          num_iterations,
                          num_warmup_iterations,
                          pool_size,
                          warmup_arc_batchsize,
                          exploit_execute_rounds)
                  +"="*10)
            for _ in tqdm(range(warmup_arc_batchsize),bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:30}{r_bar}'):
                k, node = controller.randomSample()
                #k, node = controller.uctSample(uct_sample_policy)
                arcs.append(k)
                nodes.append(node)
                r = controller.simulationWithSuperCircuitParamsAndK(k, params)
                r = penalty_function(r, node) if penalty_function is not None else r
                controller.backPropagate(node, r)
            """
            reward_k_node_list = Parallel(n_jobs=-1, verbose=0)(
                delayed(getSimulationReward)(controller, k, node, params) for k, node in zip(arcs, nodes))
            for r, _, node in reward_k_node_list:
                r = penalty_function(r, node) if penalty_function is not None else r
                controller.backPropagate(node, r)
                print("BPed!")
            """
            print("Batch Training, Size = {}, Update the Parameter Pool for One Iteration".format(warmup_arc_batchsize))
        else:
            print("=" * 10 + "Model:{}, Searching at Epoch {}/{}, Pool Size: {}, "
                             "Arc Batch Size: {}, Search Sampling Rounds: {}, Exploiting Rounds: {}"
                  .format(model.name,
                      epoch + 1,
                          num_iterations,
                          pool_size,
                          search_arc_batchsize,
                          sampling_execute_rounds,
                          exploit_execute_rounds)
                  + "=" * 10)
            #TODO: No reset? Reset will lose all the reward information obtained during the warm up stage.
            #controller._reset() # reset
            new_alpha = alpha_max - (alpha_max - alpha_min) / (num_iterations - num_warmup_iterations) * (
                        epoch + 1 - num_warmup_iterations)
            controller.alpha = new_alpha  # alpha decreases as epoch increases
            new_prune_rate = prune_constant_min + (prune_constant_max - prune_constant_min) / (
                        num_iterations - num_warmup_iterations) * (epoch + 1 - num_warmup_iterations)
            controller.prune_reward_ratio = new_prune_rate  # prune rate increases as epoch increases
            for _ in tqdm(range(search_arc_batchsize),bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:30}{r_bar}'):
                k, node = controller.sampleArcWithSuperCircParams(params)
                arcs.append(k)
                nodes.append(node)
                r = controller.simulationWithSuperCircuitParamsAndK(k, params)
                r = penalty_function(r, node) if penalty_function is not None else r
                controller.backPropagate(node, r)
            """
            reward_k_node_list = Parallel(n_jobs=-1, verbose=0)(
                delayed(getSimulationReward)(controller, k, node, params) for k, node in zip(arcs, nodes))
            for r, _, node in reward_k_node_list:
                r = penalty_function(r, node) if penalty_function is not None else r
                controller.backPropagate(node, r)
            """
            print("Batch Training, Size = {}, Update the Parameter Pool for One Iteration".format(search_arc_batchsize))
        batch_models = [model(p,c,l,k,op_pool) for k in arcs]
        batch_gradients = Parallel(n_jobs=-1, verbose=0)(
            delayed(getGradientFromModel)(constructed_model, params) for constructed_model in batch_models
        )
        #batch_gradients = [getGradientFromModel(constructed_model, params) for constructed_model in batch_models]
        batch_gradients = np.stack(batch_gradients, axis=0)
        batch_gradients = np.nan_to_num(batch_gradients)
        batch_gradients = np.mean(batch_gradients, axis=0)
        # add some noise to the circuit gradient
        noise = np.random.randn(p, c, l)
        batch_gradients = batch_gradients + noise*super_circ_train_gradient_noise_factor
        params = optimizer.apply_grad(grad=batch_gradients, args=params)
        params = np.array(params)
        print("Parameters Updated!")
        print("Exploiting and finding the best arc...")
        current_best_arc, current_best_node = controller.exploitArcWithSuperCircParams(params)
        #current_best_reward = controller.simulationWithSuperCircuitParamsAndK(current_best_arc, params)
        current_best_model = model(p, c, l, current_best_arc, op_pool)
        current_best_loss = current_best_model.getLoss(params)
        current_best_reward = current_best_model.getReward(params)
        current_penalized_best_reward = penalty_function(current_best_reward, current_best_node) if penalty_function is not None else current_best_reward
        # current_extracted_params = [params[index] for index in current_best_model.param_indices]
        # drawer = qml.draw(current_best_circ)
        end = time.time()
        print("Prune Count: {}".format(controller.prune_counter))
        print("Current Best Reward: {} (After Penalization: {}), Current Best Loss: {}".format(current_best_reward, current_penalized_best_reward, current_best_loss))
        print("Current Best k:\n", current_best_arc)
        if verbose > 1:
            print("Current Ops:")
            print(current_best_node.state)
        else:
            print("Pool:\n {}".format(op_pool))
        print("=" * 10 + "Epoch Time: {}".format(end - start) + "=" * 10+"\n")
        best_rewards.append((current_best_arc, current_best_reward))

    return params, current_best_arc, current_best_node, current_best_reward, controller, best_rewards


def circuitModelTuning(
        initial_params:Union[np.ndarray, pnp.ndarray, Sequence]=None,
        model=None,
        num_epochs:int=None,
        k:List[int]=None,
        op_pool=None,
        opt_callable=qml.AdamOptimizer,
        lr = 0.01,
        grad_noise_factor = 1/100,
        verbose = 1
):
    p, c, l = initial_params.shape[0], initial_params.shape[1], initial_params.shape[2]
    circ = model(p, c, l, k, op_pool)
    optimizer = opt_callable(stepsize=lr)
    circ_params = initial_params
    loss_list = []
    for epoch in range(num_epochs):
        loss = circ.getLoss(circ_params)
        loss_list.append(loss)
        if verbose >= 1:
            print('Training Circuit at Epoch {}/{}; Loss: {}'.format(epoch + 1, num_epochs, loss))
        gradients = circ.getGradient(circ_params)
        gradients = np.nan_to_num(gradients)
        noise = np.random.randn(p, c, l)
        gradients = gradients + noise * grad_noise_factor
        circ_params = optimizer.apply_grad(grad=gradients, args=circ_params)
        circ_params = np.array(circ_params)
    return circ_params, loss_list
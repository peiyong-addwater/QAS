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
        self.qubit_with_actions = qubit_with_actions

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
        loss = circ.getLoss(params)
        return -loss

    def randomSample(self):
        node = self.root
        while not node.state.isTerminal():
            if not node.isFullyExpanded:
                node = self.expand(node)
            else:
                action, node = random.choice(list(node.children.items()))
        return node.state.getCurrK(), node

    def getBestChild(self, node:TreeNode, alpha, policy='local_optimal'):
        if list(node.children.values()) == 0:
            raise("Empty Actions")
        if policy == 'random':
            return random.choice(list(node.children.values()))
        best_value = float("-inf")
        best_nodes, suboptimal_nodes = [], []
        if len(node.children.values()) == 0:
            raise ValueError("No Legal Child for Node at: %s"%node.state.getCurrK())
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



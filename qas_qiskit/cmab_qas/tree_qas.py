from .standard_ops import GatePool, parameterized
from .standard_ops import QuantumGate
from .circuits import QuantumCircuit, extract_ops, QCircFromK, FiveBitCodeSearchDensityMatrixNoiseless,\
    BitFlipSearchDensityMatrixNoiseless
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

class QASState():
    def __init__(self, current_k:List[int]=[], op_pool:GatePool=None, maxDepth = 30, qubit_with_actions:Set = None):
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
        # don't want two consecutive parameterized gates or Pauli gates or CNOT gates
        if len(self.current_k)>=1 and action==self.current_k[-1]:
            if op_name in parameterized:
                return False
            if op_name in ["XGate", "YGate", "ZGate", "CXGate"]:
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
        new_state = QASState(current_k=new_path, op_pool=self.pool_obj, maxDepth=self.max_depth,
                             qubit_with_actions=new_qubit_with_actions)
        new_state.state = action
        return new_state

    def isTerminal(self):
        return self.current_depth>=self.max_depth

    def getReward(self):
        return 0.

    def __repr__(self):
        disp = ""
        for i, p in enumerate(self.current_k):
            disp = disp + "OpAtDepth: {}\tOpKey: {}\tOpName: {}\n".format(i, p, self.op_name_dict[p])
        return disp

class TreeNode():
    # https://github.com/pbsinclair42/MCTS/blob/master/mcts.py
    def __init__(self, state:QASState, parent):
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
                 model:Callable=BitFlipSearchDensityMatrixNoiseless,
                 data:Optional[List[List]] = None,
                 iteration_limit = 5,
                 alpha = 1/np.sqrt(2),
                 prune_constant = 0.9,
                 eval_expansion_factor = 100,
                 op_pool:GatePool=None,
                 init_qubit_with_actions:Set = None,
                 target_circuit_depth = 2,
                 first_policy='local_optimal',
                 second_policy='local_optimal',
                 iteration_limit_ratio = 10,
                 num_minimum_children=10):
        assert len(data) == 2
        self.model = model
        self.data_list = data
        self.input_data = data[0]
        self.target_data = data[1]
        self.iteration_limit = iteration_limit
        self.alpha = alpha
        self.prune_constant = prune_constant
        self.eval_expansion_factor = eval_expansion_factor
        self.prune_counter = 0
        self.op_pool = op_pool
        self.op_pool_dict = op_pool.pool
        self.pool_keys = list(self.op_pool_dict.keys())
        self.max_depth = target_circuit_depth
        self.init_qubit_with_actions = set() if init_qubit_with_actions is None else init_qubit_with_actions
        self.initial_state = QASState(op_pool=self.op_pool, maxDepth=self.max_depth,
                                      qubit_with_actions=self.init_qubit_with_actions)
        self.first_policy = first_policy
        self.second_policy = second_policy
        self.iteration_limit_ratio = iteration_limit_ratio
        self.num_minimum_children = num_minimum_children

    def _reset(self):
        self.root = TreeNode(self.initial_state, None)
        self.prune_counter = 0

    def setRoot(self):
        self.root = TreeNode(self.initial_state, None)

    def executeRound(self, node:TreeNode, params:np.ndarray):
        if node is None:
            node = self.root
        while not node.state.isTerminal():
            node = self.selectNode(node)
        reward = self.simulation(node.state.current_k,params)
        self.backPropagate(node, reward)
        return node

    def backPropagate(self, node:TreeNode, reward):
        while node is not None:
            node.numVisits+=1
            node.totalReward += reward
            node = node.parent

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
        raise Exception("Should never reach here")

    def getAction(self, root:TreeNode, best_child):
        for action, node in root.children.items():
            if node is best_child:
                return action

    def getBestChild(self, node:TreeNode, alpha, policy='local_optimal'):
        if list(node.children.values()) == 0:
            raise("Empty Actions")
        if policy == 'random':
            return random.choice(list(node.children.values()))
        best_value = float("-inf")
        best_nodes, suboptimal_nodes = [], []
        for child in node.children.values():
            node_value = child.totalReward/child.numVisits + alpha * np.sqrt(2*np.log(node.numVisits)/child.numVisits)
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
            node = random.choice(suboptimal_nodes)
        elif policy == 'local_optimal':
            node = random.choice(best_nodes)
        else:
            raise ValueError("No such policy: %s" % policy)
        return node

    def selectNode(self, node:TreeNode):
        if node.isFullyExpanded:
            # pruning
            node_avg_reward = node.totalReward/node.numVisits
            threshold = node_avg_reward*self.prune_constant
            pruned_key = []
            for key in list(node.children.keys()):
                child = node.children[key]
                child_avg_reward = child.totalReward/child.numVisits
                if child_avg_reward<threshold and child.numVisits>(self.iteration_limit*self.iteration_limit_ratio): # also a magic number
                    pruned_key.append(key)
            random.shuffle(pruned_key)
            for i, item in enumerate(pruned_key):
                if i >= len(pruned_key) - self.num_minimum_children: # some magic number, set a threshold for the length pruned key list
                    break
                node.children.pop(item)
                self.prune_counter += 1
            node = self.getBestChild(node, self.alpha, policy=self.first_policy)
        else:
            node = self.expand(node)
        return node


    def sampleArc(self, params:np.ndarray):
        curr = self.root
        for i in range(self.iteration_limit):
            self.executeRound(node=curr, params=params)
        while not curr.state.isTerminal():
            curr = self.getBestChild(curr, self.alpha, policy=self.first_policy)
        return curr.state.current_k, curr

    def exploitArc(self, params:np.ndarray):
        curr = self.root
        while not curr.state.isTerminal():
            # execute the round from a node multiple times, then select the best child
            for i in range(self.iteration_limit * self.eval_expansion_factor):
                self.executeRound(node=curr, params=params)
            curr = self.getBestChild(curr, 0, policy=self.second_policy)
        return curr.state.current_k, curr

    def randomSample(self):
        node = self.root
        while not node.state.isTerminal():
            if not node.isFullyExpanded:
                node = self.expand(node)
            else:
                action, node = random.choice(list(node.children.items()))
        return node.state.current_k, node

    def simulation(self, k:List[int], params:np.ndarray):
        # only evaluate circuit with parameters.
        # training needs to be performed outside the MCTS class during search
        assert len(k) == self.max_depth
        p, c, l = params.shape[0], params.shape[1], params.shape[2]
        circ = self.model(p, c, l, k, self.op_pool)
        loss = circ.get_loss(params, self.input_data, self.target_data)
        avg_fid = 1-loss
        return avg_fid














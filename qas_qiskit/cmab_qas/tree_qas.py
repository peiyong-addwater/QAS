from .standard_ops import GatePool
from .standard_ops import QuantumGate
from .circuits import QuantumCircuit
from copy import deepcopy
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

class QASState():
    def __init__(self, current_k:List[int], op_pool:GatePool=None, maxDepth = 30, qubit_with_actions:Set = None):
        self.max_depth = maxDepth
        self.current_k = current_k
        self.current_depth = len(self.current_k)
        assert self.current_depth <= self.max_depth
        self.pool_obj = op_pool
        self.op_name_dict = op_pool.pool
        self.pool_keys = op_pool.pool.keys()
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
        op = self.op_name_dict[action]
        assert len(op.keys()) == 1 # in case some weird things happen
        op_name = list(op.keys())[0]
        op_qubit = op[op_name]
        if len(self.current_k)>=1 and action==self.current_k[-1]:
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
            disp = disp + "OpNo: {}, OpKey: {}, OpName: {}".format(i, p, self.op_name_dict[p])
        return disp











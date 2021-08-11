from .standard_ops import GatePool
from .standard_ops import QuantumGate
from .circuits import QuantumCircuit
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
            return None

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








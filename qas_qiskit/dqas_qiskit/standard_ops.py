import qiskit
import numpy as np
from abc import ABC, abstractmethod
from qiskit.circuit.library import (
    CU3Gate,
    U3Gate,
    HGate,
    IGate,
    SGate,
    SdgGate,
    TGate,
    TdgGate,
    XGate,
    YGate,
    ZGate,
    CXGate,
    CZGate,
    CYGate,
    CHGate,
    SXGate
)
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

_single_qubit_gate_3_params = {"U3Gate":U3Gate}
_two_qubit_gate_3_params = {"CU3Gate":CU3Gate}

_single_qubit_gate_no_param = {"HGate":HGate, "IGate":IGate, "SGate":SGate, "SdgGate":SdgGate, "TGate":TGate,
                               "TdgGate":TdgGate, "XGate":XGate, "YGate":YGate, "ZGate":ZGate, "SXGate":SXGate}
_two_qubit_gate_no_param = {"CXGate":CXGate, "CZGate":CZGate, "CYGate":CYGate, "CHGate":CHGate}

standard_quantum_ops = {}
standard_quantum_ops.update(_single_qubit_gate_3_params)
standard_quantum_ops.update(_two_qubit_gate_3_params)
standard_quantum_ops.update(_single_qubit_gate_no_param)
standard_quantum_ops.update(_two_qubit_gate_no_param)


class Op(ABC):

    @abstractmethod
    def __str__(self):
        pass
    @abstractmethod
    def get_pos(self):
        pass


class QuantumGate(Op):
    def __init__(self, name:str, pos:Tuple, param:Optional[Sequence]=None):
        assert name in standard_quantum_ops.keys()
        if name in _two_qubit_gate_3_params.keys() or name in _two_qubit_gate_no_param.keys():
            assert len(pos) == 2
        if name in _single_qubit_gate_3_params.keys() or name in _single_qubit_gate_no_param.keys():
            assert len(pos) == 1
        if name in _single_qubit_gate_3_params.keys() or name in _two_qubit_gate_3_params.keys():
            assert len(param) == 3
        if param is None:
            self.op = standard_quantum_ops[name]()
            self.param_dim = 0
        else:
            self.op = standard_quantum_ops[name](param[0],param[1],param[2])
            self.param_dim = 3
        self.pos = pos
        self.param = param
        self.name = name

    def __str__(self):
        return "({}, {}, {})".format(self.name, list(self.pos), self.param)

    def get_pos(self):
        return list(self.pos)









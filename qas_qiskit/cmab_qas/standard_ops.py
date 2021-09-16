import json
from pprint import pprint
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
    SXGate,
    RXGate,
    RYGate,
    RZGate,
    CRXGate,
    CRYGate,
    CRZGate
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

_single_qubit_gate_1_param = {"RXGate":RXGate, "RYGate":RYGate, "RZGate":RZGate}
_two_qubit_gate_1_param = {"CRXGate":CRXGate, "CRYGate":CRYGate, "CRZGate":CRZGate}

_single_qubit_gate_no_param = {"HGate":HGate, "IGate":IGate, "SGate":SGate, "SdgGate":SdgGate, "TGate":TGate,
                               "TdgGate":TdgGate, "XGate":XGate, "YGate":YGate, "ZGate":ZGate, "SXGate":SXGate}
_two_qubit_gate_no_param = {"CXGate":CXGate, "CZGate":CZGate, "CYGate":CYGate, "CHGate":CHGate}

parameterized = {}
parameterized.update(_single_qubit_gate_3_params)
parameterized.update(_two_qubit_gate_3_params)
parameterized.update(_single_qubit_gate_1_param)
parameterized.update(_two_qubit_gate_1_param)

non_parameterized = {}
non_parameterized.update(_single_qubit_gate_no_param)
non_parameterized.update(_two_qubit_gate_no_param)


single_qubit_ops = {}
single_qubit_ops.update(_single_qubit_gate_no_param)
single_qubit_ops.update(_single_qubit_gate_3_params)
single_qubit_ops.update(_single_qubit_gate_1_param)

two_qubit_ops = {}
two_qubit_ops.update(_two_qubit_gate_no_param)
two_qubit_ops.update(_two_qubit_gate_3_params)
two_qubit_ops.update(_two_qubit_gate_1_param)

standard_quantum_ops = {}
standard_quantum_ops.update(_single_qubit_gate_3_params)
standard_quantum_ops.update(_two_qubit_gate_3_params)
standard_quantum_ops.update(_single_qubit_gate_no_param)
standard_quantum_ops.update(_two_qubit_gate_no_param)
standard_quantum_ops.update(_single_qubit_gate_1_param)
standard_quantum_ops.update(_two_qubit_gate_1_param)

def supported_ops():
    pprint(standard_quantum_ops)

class Op(ABC):

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def get_qreg_pos(self):
        pass

    @abstractmethod
    def get_creg_pos(self):
        pass



class QuantumGate(Op):
    def __init__(self, name:str, pos:Tuple[int], param:Optional[Sequence]=None):
        assert name in standard_quantum_ops.keys()
        if name in _two_qubit_gate_3_params.keys() or name in _two_qubit_gate_no_param.keys():
            assert len(pos) == 2
        if name in _single_qubit_gate_3_params.keys() or name in _single_qubit_gate_no_param.keys():
            assert len(pos) == 1
        if name in parameterized.keys():
            assert len(param) == 3
            self.param_dim = 3
            self.param = param
            if name in _single_qubit_gate_3_params.keys() or name in _two_qubit_gate_3_params.keys():
                self.op = standard_quantum_ops[name](param[0], param[1], param[2])
            if name in _single_qubit_gate_1_param.keys() or name in _two_qubit_gate_1_param.keys():
                self.op = standard_quantum_ops[name](param[0])
        else:
            self.op = standard_quantum_ops[name]()
            self.param_dim = 0
            self.param = None
        self.pos = pos
        self.name = name

    def __str__(self):
        if self.param is not None:
            return "({}, {}, {})".format(self.name, list(self.pos), list(self.param))
        else:
            return "({}, {}, {})".format(self.name, list(self.pos), None)

    def get_op(self):
        return self.op

    def get_qreg_pos(self):
        return list(self.pos)

    def get_creg_pos(self):
        #TODO: Gates with classical infomation as control
        raise NotImplementedError



class GatePool(dict):
    def __init__(self, num_qubits:int, single_qubit_op_generator:List[AnyStr],
                 two_qubit_op_generator:List[AnyStr],
                 complete_undirected_graph:bool=True, two_qubit_gate_map:Optional[List[Tuple[int,int]]]=None):
        super(GatePool, self).__init__()
        for c in single_qubit_op_generator:
            assert c in single_qubit_ops.keys()
        for c in two_qubit_op_generator:
            assert c in two_qubit_ops.keys()
        if two_qubit_gate_map is not None:
            assert complete_undirected_graph is False
            self.coupling = two_qubit_gate_map
            for c in self.coupling:
                assert len(c) == 2
                assert c[0] != c[1]
                assert c[0] >= 0
                assert c[0] < num_qubits
                assert c[1] >= 0
                assert c[1] < num_qubits
        else:
            self.coupling = []
            for i in range(num_qubits):
                for j in range(num_qubits):
                    if i!=j:
                        self.coupling.append((i,j))

        self.num_qubits = num_qubits
        self.single_qubit_gates = single_qubit_op_generator
        self.two_qubit_gates = two_qubit_op_generator
        self.pool = {}
        pool_key = 0
        for i in range(num_qubits):
            for c in single_qubit_op_generator:
                self.pool[pool_key] = {c:(i,)}
                pool_key = pool_key+1
        for couple_direction in self.coupling:
            for c in two_qubit_op_generator:
                self.pool[pool_key] = {c:couple_direction}
                pool_key = pool_key + 1

    def __getitem__(self, key):
        return self.pool[key]

    def __str__(self):
        return json.dumps(self.pool)

    def __len__(self):
        return len(self.pool.keys())

def default_complete_graph_parameterized_pool(num_qubits:int)->GatePool:
    s = ["U3Gate"]
    d = ["CU3Gate"]
    return GatePool(num_qubits,s,d)

def default_complete_graph_non_parameterized_pool(num_qubits:int)->GatePool:
    d = ["CXGate"]
    s = ["XGate", "YGate", "ZGate", "HGate", "TGate", "SGate", "HGate", "SXGate", "TdgGate", "SdgGate", "IGate"]
    return GatePool(num_qubits, s, d)

def simple_complete_graph_non_parameterized_pool(num_qubits:int)->GatePool:
    d = ["CXGate"]
    s = ["XGate", "YGate", "ZGate", "HGate","TGate","TdgGate"]
    return GatePool(num_qubits, s, d)




from .models import ModelFromK
from .qml_ops import (
    SUPPORTED_OPS_DICT,
    SUPPORTED_OPS_NAME,
    QMLPool,
    QMLGate
)
import pennylane.numpy as pnp
import numpy as np
import jax.numpy as jnp
import pennylane as qml
import jax
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

ket0 = np.array([1, 0])
ket1 = np.array([0, 1])
PAULI_EIGENSTATES_T_STATE = [(ket0 + ket1) / np.sqrt(2), (ket0 - ket1) / np.sqrt(2), ket0, ket1,
                             (ket0 + ket1 * 1j) / np.sqrt(2), (ket0 - ket1 * 1j) / np.sqrt(2),
                             (ket0 + np.exp(np.pi * 1j / 4) * ket1) / np.sqrt(2)]

STATE_DIC = {'|0>': ket0, '|1>': ket1, '|+>': (ket0 + ket1) / np.sqrt(2), '|->': (ket0 - ket1) / np.sqrt(2),
             '|+i>': (ket0 + ket1 * 1j) / np.sqrt(2), '|-i>': (ket0 - ket1 * 1j) / np.sqrt(2),
             'magic': (ket0 + np.exp(np.pi * 1j / 4) * ket1) / np.sqrt(2)}
STATE_DICT_KEYS = list(STATE_DIC.keys())

def extractParamIndices(k:List[int], op_pool:QMLPool, circ_params:Union[np.ndarray, jnp.ndarray, pnp.ndarray, Sequence])->List:
    p, c, l = circ_params.shape[0], circ_params.shape[1], circ_params.shape[2]
    assert min(k) >= 0
    assert max(k) < c
    assert p == len(k)
    assert c == len(op_pool)
    param_indices = []
    for i in range(p):
        gate_dict = op_pool[k[i]]
        assert len(gate_dict.keys()) == 1
        gate_name = list(gate_dict.keys())[0]
        gate_obj = SUPPORTED_OPS_DICT[gate_name]
        num_params = gate_obj.num_params
        if num_params > 0:
            for j in range(num_params):
                param_indices.append((i, k[i], j))
    return param_indices

def extractGates(k:List[int], op_pool:QMLPool, circ_params:Union[np.ndarray, jnp.ndarray, pnp.ndarray, Sequence])->List:
    p, c, l = circ_params.shape[0], circ_params.shape[1], circ_params.shape[2]
    assert min(k) >= 0
    assert max(k) < c
    assert p == len(k)
    assert c == len(op_pool)
    extracted_gates = []
    for i in range(p):
        gate_info = op_pool[k[i]]
        assert len(gate_info.keys()) == 1
        gate_name = list(gate_info.keys())[0]
        gate_obj = SUPPORTED_OPS_DICT[gate_name]
        num_params = gate_obj.num_params
        gate_pos = gate_info[gate_name]
        gate_param = circ_params[i, k[i], :] if num_params > 0 else None
        qml_gate = QMLGate(gate_name, gate_pos, gate_param)
        extracted_gates.append(qml_gate)
    return extracted_gates

FOUR_TWO_TWO_DETECTION_CODE_INPUT = []
for c in PAULI_EIGENSTATES_T_STATE:
    for d in PAULI_EIGENSTATES_T_STATE:
        FOUR_TWO_TWO_DETECTION_CODE_INPUT.append([c,d])

TOFFOLI_INPUT = []
for a in [ket0, ket1]:
    for b in [ket0, ket1]:
        for c in [ket0, ket1]:
            temp = np.kron(a, b)
            s = np.kron(temp, c)
            TOFFOLI_INPUT.append(s)



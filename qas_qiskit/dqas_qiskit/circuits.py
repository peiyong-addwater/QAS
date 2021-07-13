import qiskit
from qiskit import QuantumCircuit
import numpy as np
import jax.numpy as jnp
import jax
import optax
import functools
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
from .standard_ops import *

ket0 = np.array([1, 0])
ket1 = np.array([0, 1])
PAULI_EIGENSTATES_T_STATE = [(ket0 + ket1) / np.sqrt(2), (ket0 - ket1) / np.sqrt(2), ket0, ket1,
                             (ket0 + ket1 * 1j) / np.sqrt(2), (ket0 - ket1 * 1j) / np.sqrt(2),
                             (ket0 + np.exp(np.pi * 1j / 4) * ket1) / np.sqrt(2)]

STATE_DIC = {'|0>': ket0, '|1>': ket1, '|+>': (ket0 + ket1) / np.sqrt(2), '|->': (ket0 - ket1) / np.sqrt(2),
             '|+i>': (ket0 + ket1 * 1j) / np.sqrt(2), '|-i>': (ket0 - ket1 * 1j) / np.sqrt(2),
             'magic': (ket0 + np.exp(np.pi * 1j / 4) * ket1) / np.sqrt(2)}
STATE_DICT_KEYS = list(STATE_DIC.keys())

def construct_circuit(num_qubits:int, k:List[int],op_pool:GatePool, circ_params:jnp.ndarray)->(QuantumCircuit, List[AnyStr]):
    p, c, l = circ_params.shape[0], circ_params.shape[1], circ_params.shape[2]
    assert p == len(k)
    assert c == len(op_pool)
    assert l == 3 # force max num of parameters equal to 3


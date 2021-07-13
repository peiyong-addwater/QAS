import qiskit
from qiskit.circuit.library import (
    CUGate,
    CU3Gate,
    UGate,
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
)

_single_qubit_gate_3_params = {"U3Gate":U3Gate}
_two_qubit_gate_3_params = {"CU3Gate":CU3Gate}

_single_qubit_gate_no_param = {"HGate":HGate, "IGate":IGate, "SGate":SGate, "SdgGate":SdgGate, "TGate":TGate,
                               "TdgGate":TdgGate, "XGate":XGate, "YGate":YGate, "ZGate":ZGate, "CXGate":CXGate,
                               "CZGate":CZGate, "CYGate":CYGate, "CHGate":CHGate, "SXGate":SXGate}


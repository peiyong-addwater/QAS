from qas.models import ModelFromK
from qas.qml_gate_ops import (
    SUPPORTED_OPS_DICT,
    SUPPORTED_OPS_NAME,
    QMLPool,
    QMLGate
)
import pennylane.numpy as pnp
from pennylane.operation import Operation, AnyWires
from pennylane import numpy as np
import pennylane as qml
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

from collections import OrderedDict, Counter
import shutup
shutup.please()
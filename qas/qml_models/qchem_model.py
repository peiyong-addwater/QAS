# Some models for quantum chemistry problems using Pennylane's quantum dataset
# https://pennylane.ai/qml/datasets_qchem.html
from qas.qml_models.utils import extractParamIndicesQML
from qas.qml_models.qml_gate_ops import QMLPool, QMLGate, SUPPORTED_OPS_DICT, SUPPORTED_OPS_NAME
import pennylane as qml
from pennylane import numpy as pnp
import numpy as np
from qas.models import ModelFromK
from typing import (
    List,
    Sequence,
    Union
)
import shutup
shutup.please()

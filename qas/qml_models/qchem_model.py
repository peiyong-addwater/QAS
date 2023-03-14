# Some models for quantum chemistry problems using Pennylane's quantum dataset
# https://pennylane.ai/qml/datasets_qchem.html
from qas.qml_models.utils import extractParamIndicesQML
from qas.qml_models.hamiltonian_model import HamiltonianModel
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

H2_STO_3G_DATA = qml.data.load("qchem", molname="H2", basis="STO-3G", bondlength=0.742)[0]
LiH_STO_3G_DATA = qml.data.load("qchem", molname="LiH", basis="STO-3G", bondlength=1.57)[0]
H2O_STO_3G_DATA = qml.data.load("qchem", molname="H2O", basis="STO-3G", bondlength=0.958)[0]


class H2STO3G(HamiltonianModel):
    name = "H2_STO-3G"
    def __init__(self, p: int, c: int, l: int, structure_list: List[int], op_pool: Union[QMLPool, dict]):
        super().__init__(p, c, l, structure_list, op_pool)
        self.k = structure_list
        self.pool = op_pool
        self.p, self.c, self.l = p, c, l
        self.num_qubits = 4
        self.param_indices = extractParamIndicesQML(self.k, self.pool)
        self.dev = qml.device("lightning.qubit", wires=self.num_qubits, batch_obs=True)
        self.H = H2_STO_3G_DATA.hamiltonian
        self.fci_energy = H2_STO_3G_DATA.fci_energy

class LiHSTO3G(HamiltonianModel):
    name = "LiH_STO-3G"
    def __init__(self, p: int, c: int, l: int, structure_list: List[int], op_pool: Union[QMLPool, dict]):
        super().__init__(p, c, l, structure_list, op_pool)
        self.k = structure_list
        self.pool = op_pool
        self.p, self.c, self.l = p, c, l
        self.num_qubits = 12
        self.param_indices = extractParamIndicesQML(self.k, self.pool)
        self.dev = qml.device("lightning.qubit", wires=self.num_qubits, batch_obs=True)
        self.H = LiH_STO_3G_DATA.hamiltonian
        self.fci_energy = LiH_STO_3G_DATA.fci_energy

class H2OSTO3G(HamiltonianModel):
    name = "H2O_STO-3G"
    def __init__(self, p: int, c: int, l: int, structure_list: List[int], op_pool: Union[QMLPool, dict]):
        super().__init__(p, c, l, structure_list, op_pool)
        self.k = structure_list
        self.pool = op_pool
        self.p, self.c, self.l = p, c, l
        self.num_qubits = 14
        self.param_indices = extractParamIndicesQML(self.k, self.pool)
        self.dev = qml.device("lightning.qubit", wires=self.num_qubits, batch_obs=True)
        self.H = H2O_STO_3G_DATA.hamiltonian
        self.fci_energy = H2O_STO_3G_DATA.fci_energy

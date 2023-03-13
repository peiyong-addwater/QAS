# Model parameters from https://github.com/qiskit-community/open-science-prize-2022/blob/main/kagome-vqe.ipynb
# A 12-site kagome lattice on a 16-qubit quantum processor
# The Hamiltonian is
# H = \\sum_{\\langle i j \\rangle}^N X_i X_{j} + Y_i Y_{j} + Z_i Z_{j}

from qas.qml_models.utils import extractParamIndicesQML
from qas.qml_models.qml_gate_ops import QMLPool, QMLGate, SUPPORTED_OPS_DICT, SUPPORTED_OPS_NAME
import pennylane as qml
import numpy as np
from pennylane import numpy as pnp
from qas.models import ModelFromK
from typing import (
    List,
    Sequence,
    Union
)
import shutup
shutup.please()

t = 1.0
kagome_edge_list = [
    (1, 2, t),
    (2, 3, t),
    (3, 5, t),
    (5, 8, t),
    (8, 11, t),
    (11, 14, t),
    (14, 13, t),
    (13, 12, t),
    (12, 10, t),
    (10, 7, t),
    (7, 4, t),
    (4, 1, t),
    (4, 2, t),
    (2, 5, t),
    (5, 11, t),
    (11, 13, t),
    (13, 10, t),
    (10, 4, t),
]

kagome_coeffs = []
kagome_obs = []

for edge in kagome_edge_list:
    kagome_coeffs.append(t)
    kagome_obs.append(qml.PauliX(edge[0])@qml.PauliX(edge[1]))
    kagome_coeffs.append(t)
    kagome_obs.append(qml.PauliY(edge[0]) @ qml.PauliY(edge[1]))
    kagome_coeffs.append(t)
    kagome_obs.append(qml.PauliZ(edge[0]) @ qml.PauliZ(edge[1]))

kagome_hamiltonian = qml.Hamiltonian(kagome_coeffs, kagome_obs)

class KagomeHeisenberg(ModelFromK):

    name = "KagomeHeisenberg"

    def __init__(self, p: int, c: int, l: int, structure_list: List[int], op_pool: Union[QMLPool, dict]):
        self.k = structure_list
        self.pool = op_pool
        self.p, self.c, self.l = p, c, l
        self.num_qubits = 16
        self.param_indices = extractParamIndicesQML(self.k, self.pool)
        self.dev = qml.device("lightning.gpu", wires=self.num_qubits, batch_obs=True)
        self.H = kagome_hamiltonian # ground state energy -18, two ground states

    def backboneCirc(self, extracted_params):
        param_pos = 0
        for i in range(self.p):
            gate_dict = self.pool[self.k[i]]
            assert len(gate_dict.keys()) == 1
            gate_name = list(gate_dict.keys())[0]
            if gate_name != "PlaceHolder":
                gate_obj = SUPPORTED_OPS_DICT[gate_name]
                num_params = gate_obj.num_params
                wires = gate_dict[gate_name]
                if num_params > 0:
                    gate_params = []
                    for j in range(num_params):
                        gate_params.append(extracted_params[param_pos])
                        param_pos = param_pos + 1
                    qml_gate_obj = QMLGate(gate_name, wires, gate_params)
                else:
                    gate_params = None
                    qml_gate_obj = QMLGate(gate_name, wires, gate_params)
                qml_gate_obj.getOp()

    def constructFullCirc(self):
        @qml.qnode(self.dev, diff_method="adjoint")
        def fullCirc(extracted_params):
            # qml.BasisState(self.hf, wires=[0,1,2,3])
            self.backboneCirc(extracted_params)
            return qml.expval(self.H)
        return fullCirc

    def costFunc(self, extracted_params):
        circ_func = self.constructFullCirc()
        return circ_func(extracted_params)

    def getLoss(self, super_circ_params: Union[np.ndarray, pnp.ndarray, Sequence]):
        assert super_circ_params.shape[0] == self.p
        assert super_circ_params.shape[1] == self.c
        assert super_circ_params.shape[2] == self.l
        extracted_params = []
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        extracted_params = np.array(extracted_params)
        return self.costFunc(extracted_params)

    def getReward(self, super_circ_params: Union[np.ndarray, pnp.ndarray, Sequence]):
        assert super_circ_params.shape[0] == self.p
        assert super_circ_params.shape[1] == self.c
        assert super_circ_params.shape[2] == self.l
        extracted_params = []
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        extracted_params = np.array(extracted_params)
        return -self.costFunc(extracted_params)

    def getGradient(self, super_circ_params: Union[np.ndarray, pnp.ndarray, Sequence]):
        assert super_circ_params.shape[0] == self.p
        assert super_circ_params.shape[1] == self.c
        assert super_circ_params.shape[2] == self.l
        extracted_params = []
        gradients = np.zeros(super_circ_params.shape)
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        extracted_params = np.array(extracted_params, requires_grad=True)  # needed for the new pennylane version
        if len(extracted_params) == 0:
            return gradients
        cost_grad = qml.grad(self.costFunc)
        extracted_gradients = cost_grad(extracted_params)
        for i in range(len(self.param_indices)):
            gradients[self.param_indices[i]] = extracted_gradients[i]

        return gradients

    def toList(self, super_circ_params):
        extracted_params = []
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        gate_list = []
        param_pos = 0
        for i in self.k:
            gate_dict = self.pool[i]
            assert len(gate_dict.keys()) == 1
            gate_name = list(gate_dict.keys())[0]
            if gate_name != "PlaceHolder":
                gate_pos = gate_dict[gate_name]
                gate_obj = SUPPORTED_OPS_DICT[gate_name]
                gate_num_params = gate_obj.num_params
                if gate_num_params > 0:
                    gate_params = []
                    for j in range(gate_num_params):
                        gate_params.append(extracted_params[param_pos])
                        param_pos = param_pos + 1
                    gate_list.append((gate_name, gate_pos, gate_params))
                else:
                    gate_list.append((gate_name, gate_pos, None))

        return gate_list

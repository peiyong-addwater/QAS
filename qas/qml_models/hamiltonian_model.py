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

class HamiltonianModel(ModelFromK):
    name: str
    def __init__(self, p: int, c: int, l: int, structure_list: List[int], op_pool: Union[QMLPool, dict]):
        self.k = structure_list
        self.pool = op_pool
        self.p, self.c, self.l = p, c, l
        self.param_indices = extractParamIndicesQML(self.k, self.pool)
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
        extracted_params = pnp.array(extracted_params, requires_grad=True)  # needed for the new pennylane version
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
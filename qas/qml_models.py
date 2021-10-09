from .models import ModelFromK
from .qml_gate_ops import (
    SUPPORTED_OPS_DICT,
    SUPPORTED_OPS_NAME,
    QMLPool,
    QMLGate
)
import pennylane.numpy as pnp
from pennylane.operation import Operation, AnyWires
import numpy as np
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
import qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import state_fidelity, DensityMatrix, Statevector
from qiskit.providers.aer.backends import StatevectorSimulator, AerSimulator, QasmSimulator

ket0 = np.array([1, 0])
ket1 = np.array([0, 1])
PAULI_EIGENSTATES_T_STATE = [(ket0 + ket1) / np.sqrt(2), (ket0 - ket1) / np.sqrt(2), ket0, ket1,
                             (ket0 + ket1 * 1j) / np.sqrt(2), (ket0 - ket1 * 1j) / np.sqrt(2),
                             (ket0 + np.exp(np.pi * 1j / 4) * ket1) / np.sqrt(2)]

PAULI_X = np.array([[0,1],[1,0]])
PAULI_Z = np.array(([[1,0],[0,-1]]))
PAULI_Y = -1j*np.matmul(PAULI_X, PAULI_Z)
IDENTITY = np.array([[1,0],[0,1]])

STATE_DIC = {'|0>': ket0, '|1>': ket1, '|+>': (ket0 + ket1) / np.sqrt(2), '|->': (ket0 - ket1) / np.sqrt(2),
             '|+i>': (ket0 + ket1 * 1j) / np.sqrt(2), '|-i>': (ket0 - ket1 * 1j) / np.sqrt(2),
             'magic': (ket0 + np.exp(np.pi * 1j / 4) * ket1) / np.sqrt(2)}
STATE_DICT_KEYS = list(STATE_DIC.keys())

FOUR_TWO_TWO_DETECTION_CODE_INPUT = []
for c in PAULI_EIGENSTATES_T_STATE:
    for d in PAULI_EIGENSTATES_T_STATE:
        FOUR_TWO_TWO_DETECTION_CODE_INPUT.append(np.kron(c,d))

TOFFOLI_INPUT = []
EXTENDED_TOFFILI_INPUT = []
for a in [ket0, ket1]:
    for b in [ket0, ket1]:
        for c in [ket0, ket1]:
            temp = np.kron(a, b)
            s = np.kron(temp, c)
            TOFFOLI_INPUT.append(s)
            EXTENDED_TOFFILI_INPUT.append(s)
TWO_QUBIT_ENTANGLED_STATES = []
TWO_QUBIT_ENTANGLED_STATES.append(1/np.sqrt(2)*(np.kron(ket0, ket0)+np.kron(ket1, ket1)))
TWO_QUBIT_ENTANGLED_STATES.append(1/np.sqrt(2)*(np.kron(ket0, ket0)-np.kron(ket1, ket1)))
TWO_QUBIT_ENTANGLED_STATES.append(1/np.sqrt(2)*(np.kron(ket1, ket0)+np.kron(ket0, ket1)))
TWO_QUBIT_ENTANGLED_STATES.append(1/np.sqrt(2)*(np.kron(ket1, ket0)-np.kron(ket0, ket1)))
TWO_QUBIT_ENTANGLED_STATES.append(1/np.sqrt(2)*(np.kron(ket0, ket0)+1j*np.kron(ket1, ket1)))
TWO_QUBIT_ENTANGLED_STATES.append(1/np.sqrt(2)*(np.kron(ket0, ket0)-1j*np.kron(ket1, ket1)))
TWO_QUBIT_ENTANGLED_STATES.append(1/np.sqrt(2)*(np.kron(ket1, ket0)+1j*np.kron(ket0, ket1)))
TWO_QUBIT_ENTANGLED_STATES.append(1/np.sqrt(2)*(np.kron(ket1, ket0)-1j*np.kron(ket0, ket1)))
for c in TWO_QUBIT_ENTANGLED_STATES:
    for d in PAULI_EIGENSTATES_T_STATE:
        s = np.kron(c, d)
        s1 = np.kron(d, c)
        EXTENDED_TOFFILI_INPUT.append(s)
        EXTENDED_TOFFILI_INPUT.append(s1)

THREE_QUBIT_GHZ_STATES = []
THREE_QUBIT_GHZ_STATES.append(1/np.sqrt(2)*(np.kron(ket0, np.kron(ket0, ket0))+np.kron(ket1, np.kron(ket1, ket1))))
THREE_QUBIT_GHZ_STATES.append(1/np.sqrt(2)*(np.kron(ket0, np.kron(ket0, ket0))-np.kron(ket1, np.kron(ket1, ket1))))
THREE_QUBIT_GHZ_STATES.append(1/np.sqrt(2)*(np.kron(ket0, np.kron(ket0, ket0))+1j*np.kron(ket1, np.kron(ket1, ket1))))
THREE_QUBIT_GHZ_STATES.append(1/np.sqrt(2)*(np.kron(ket0, np.kron(ket0, ket0))-1j*np.kron(ket1, np.kron(ket1, ket1))))
EXTENDED_TOFFILI_INPUT.extend(THREE_QUBIT_GHZ_STATES)
FOUR_TWO_TWO_DETECTION_CODE_INPUT.extend(TWO_QUBIT_ENTANGLED_STATES)

SIMPLE_PHASE_FLIP_DATA = []
for x in PAULI_EIGENSTATES_T_STATE:
    dev_pf = qml.device('default.qubit', wires=3)
    @qml.qnode(dev_pf)
    def phase_flip_circ(x):
        qml.QubitStateVector(x, wires=[0])
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0,1])
        qml.CNOT(wires=[0,2])
        return qml.density_matrix(wires=[0,1,2])
    SIMPLE_PHASE_FLIP_DATA.append((x, phase_flip_circ(x)))

TOFFOLI_DATA = []
for x in TOFFOLI_INPUT:
    dev_t = qml.device('default.qubit', wires=3)
    @qml.qnode(dev_t)
    def toffoli_circ(x):
        qml.QubitStateVector(x, wires=[0,1,2])
        qml.Toffoli(wires=[0,1,2])
        return qml.density_matrix(wires=[0, 1, 2])
    TOFFOLI_DATA.append((x, toffoli_circ(x)))


def extractParamIndicesQML(k:List[int], op_pool:Union[QMLPool, dict])->List:
    assert min(k) >= 0
    p = len(k)
    c = len(op_pool)
    assert max(k) < c
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

def generate_single_qubit_state(theta_B:float, phi_B:float)->np.ndarray:
    return np.cos(theta_B/2)*ket0 + np.sin(theta_B/2)*np.exp(1j*phi_B)*ket1

class PhaseFlipQMLNoiseless(ModelFromK):
    name = "PhaseFlipQMLNoiseless"
    def __init__(self, p:int, c:int, l:int, structure_list:List[int], op_pool:Union[QMLPool, dict]):
        self.k = structure_list
        self.pool = op_pool
        self.p, self.c, self.l = p, c, l
        self.num_qubits = 3
        self.param_indices = extractParamIndicesQML(self.k, self.pool)
        self.data = SIMPLE_PHASE_FLIP_DATA
        self.dev = qml.device('default.qubit', wires=3)

    @qml.template
    def backboneCirc(self, extracted_params):
        param_pos = 0
        for i in range(self.p):
            gate_dict = self.pool[self.k[i]]
            assert len(gate_dict.keys()) == 1
            gate_name = list(gate_dict.keys())[0]
            if gate_name !="PlaceHolder":
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
        @qml.qnode(self.dev)
        def fullCirc(extracted_params, x=None, y = None):
            qml.QubitStateVector(x, wires=[0])
            self.backboneCirc(extracted_params)
            return qml.expval(qml.Hermitian(y, wires=[0,1,2]))
        return fullCirc

    def costFunc(self, extracted_params):
        fid = 0
        circ_func = self.constructFullCirc()
        num_data = len(self.data)
        for i in range(num_data):
            fid = fid + circ_func(extracted_params, x=self.data[i][0], y=self.data[i][1])
        return 1 - fid/num_data

    def getLoss(self, super_circ_params:Union[np.ndarray, pnp.ndarray, Sequence]):
        assert super_circ_params.shape[0] == self.p
        assert super_circ_params.shape[1] == self.c
        assert super_circ_params.shape[2] == self.l
        extracted_params = []
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        extracted_params = np.array(extracted_params)
        return self.costFunc(extracted_params)

    def getReward(self, super_circ_params:Union[np.ndarray, pnp.ndarray, Sequence]):
        assert super_circ_params.shape[0] == self.p
        assert super_circ_params.shape[1] == self.c
        assert super_circ_params.shape[2] == self.l
        extracted_params = []
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        extracted_params = np.array(extracted_params)
        return 1-self.costFunc(extracted_params)


    def getGradient(self, super_circ_params:Union[np.ndarray, pnp.ndarray, Sequence]):
        assert super_circ_params.shape[0] == self.p
        assert super_circ_params.shape[1] == self.c
        assert super_circ_params.shape[2] == self.l
        extracted_params = []
        gradients = np.zeros(super_circ_params.shape)
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])

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

class ToffoliQMLNoiseless(ModelFromK):
    name = "ToffoliQMLNoiseless"
    def __init__(self, p:int, c:int, l:int, structure_list:List[int], op_pool:Union[QMLPool, dict]):
        self.k = structure_list
        self.pool = op_pool
        self.p, self.c, self.l = p, c, l
        self.num_qubits = 3
        self.param_indices = extractParamIndicesQML(self.k, self.pool)
        self.data = TOFFOLI_DATA
        self.dev = qml.device('default.qubit', wires = self.num_qubits)
        #self.dev = qml.device('qiskit.aer', wires=self.num_qubits, max_parallel_threads=0, max_parallel_experiments=0)

    @qml.template
    def backboneCirc(self, extracted_params):
        param_pos = 0
        for i in range(self.p):
            gate_dict = self.pool[self.k[i]]
            assert len(gate_dict.keys()) == 1
            gate_name = list(gate_dict.keys())[0]
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
        @qml.qnode(self.dev)
        def fullCirc(extracted_params, x=None, y = None):
            qml.QubitStateVector(x, wires=[0,1,2])
            self.backboneCirc(extracted_params)
            return qml.expval(qml.Hermitian(y, wires=[0,1,2]))
        return fullCirc

    def costFunc(self, extracted_params):
        fid = 0
        circ_func = self.constructFullCirc()
        num_data = len(self.data)
        for c in self.data:
            fid = fid + circ_func(extracted_params, x=c[0], y=c[1])
        return 1 - fid/num_data

    def getLoss(self, super_circ_params:Union[np.ndarray, pnp.ndarray, Sequence]):
        assert super_circ_params.shape[0] == self.p
        assert super_circ_params.shape[1] == self.c
        assert super_circ_params.shape[2] == self.l
        extracted_params = []
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        extracted_params = np.array(extracted_params)
        return self.costFunc(extracted_params)

    def getReward(self, super_circ_params:Union[np.ndarray, pnp.ndarray, Sequence]):
        assert super_circ_params.shape[0] == self.p
        assert super_circ_params.shape[1] == self.c
        assert super_circ_params.shape[2] == self.l
        extracted_params = []
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        extracted_params = np.array(extracted_params)
        return 1-self.costFunc(extracted_params)


    def getGradient(self, super_circ_params:Union[np.ndarray, pnp.ndarray, Sequence]):
        assert super_circ_params.shape[0] == self.p
        assert super_circ_params.shape[1] == self.c
        assert super_circ_params.shape[2] == self.l
        extracted_params = []
        gradients = np.zeros(super_circ_params.shape)
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])

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

class ToffoliQMLSwapTestNoiseless(ModelFromK):
    name = "ToffoliQMLSwapTestNoiseless"
    def __init__(self, p:int, c:int, l:int, structure_list:List[int], op_pool:Union[QMLPool, dict]):
        self.k = structure_list
        self.pool = op_pool
        self.p, self.c, self.l = p, c, l
        self.num_qubits = 3
        self.param_indices = extractParamIndicesQML(self.k, self.pool)
        self.x_list = TOFFOLI_INPUT
        self.dev = qml.device('default.qubit', wires = self.num_qubits*2+1)
        #self.dev = qml.device('qiskit.aer', wires=self.num_qubits, max_parallel_threads=0, max_parallel_experiments=0)

    @qml.template
    def backboneCirc(self, extracted_params):
        param_pos = 0
        for i in range(self.p):
            gate_dict = self.pool[self.k[i]]
            assert len(gate_dict.keys()) == 1
            gate_name = list(gate_dict.keys())[0]
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
        @qml.qnode(self.dev)
        def fullCirc(extracted_params, x=None):

            qml.QubitStateVector(np.kron(x,x), wires=[0,1,2,3,4,5])

            self.backboneCirc(extracted_params)
            qml.Toffoli(wires=[3,4,5])
            qml.Hadamard(wires=6)
            qml.CSWAP(wires=[6,5,2])
            qml.CSWAP(wires=[6,4,1])
            qml.CSWAP(wires=[6,3,0])
            qml.Hadamard(wires=6)
            return qml.expval(qml.Hermitian(np.outer(ket0, ket0), wires=6))
        return fullCirc

    def costFunc(self, extracted_params):
        circ_func = self.constructFullCirc()
        fid = 0
        for i in range(len(self.x_list)):
            fid = fid +  (circ_func(extracted_params, x=self.x_list[i])-1/2)*2
        return 1 - fid/len(self.x_list)

    def getLoss(self, super_circ_params:Union[np.ndarray, pnp.ndarray, Sequence]):
        assert super_circ_params.shape[0] == self.p
        assert super_circ_params.shape[1] == self.c
        assert super_circ_params.shape[2] == self.l
        extracted_params = []
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        extracted_params = np.array(extracted_params)
        return self.costFunc(extracted_params)

    def getReward(self, super_circ_params:Union[np.ndarray, pnp.ndarray, Sequence]):
        assert super_circ_params.shape[0] == self.p
        assert super_circ_params.shape[1] == self.c
        assert super_circ_params.shape[2] == self.l
        extracted_params = []
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        extracted_params = np.array(extracted_params)
        return 1-self.costFunc(extracted_params)


    def getGradient(self, super_circ_params:Union[np.ndarray, pnp.ndarray, Sequence]):
        assert super_circ_params.shape[0] == self.p
        assert super_circ_params.shape[1] == self.c
        assert super_circ_params.shape[2] == self.l
        extracted_params = []
        gradients = np.zeros(super_circ_params.shape)
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])

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

class FourTwoTwoQMLNoiseless(ModelFromK):
    name="FourTwoTwoQMLNoiseless"
    def __init__(self, p:int, c:int, l:int, structure_list:List[int], op_pool:Union[QMLPool, dict]):
        self.k = structure_list
        self.pool = op_pool
        self.p, self.c, self.l = p, c, l
        self.num_qubits = 4
        self.param_indices = extractParamIndicesQML(self.k, self.pool)
        self.x_list = FOUR_TWO_TWO_DETECTION_CODE_DATA[0]
        self.y_list = FOUR_TWO_TWO_DETECTION_CODE_DATA[1]
        self.dev = qml.device('default.qubit', wires = self.num_qubits)

    @qml.template
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
        @qml.qnode(self.dev)
        def fullCirc(extracted_params, x=None, y = None):
            qml.QubitStateVector(x, wires=[0,1])
            self.backboneCirc(extracted_params)
            return qml.expval(qml.Hermitian(y, wires=[0,1,2,3]))
        return fullCirc

    def costFunc(self, extracted_params):
        fid = 0
        circ_func = self.constructFullCirc()
        num_data = len(self.x_list)
        for i in range(num_data):
            fid = fid + circ_func(extracted_params, x=self.x_list[i], y=self.y_list[i])
        return 1-fid/num_data

    def getLoss(self, super_circ_params:Union[np.ndarray, pnp.ndarray, Sequence]):
        assert super_circ_params.shape[0] == self.p
        assert super_circ_params.shape[1] == self.c
        assert super_circ_params.shape[2] == self.l
        extracted_params = []
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        extracted_params = np.array(extracted_params)
        return self.costFunc(extracted_params)

    def getReward(self, super_circ_params:Union[np.ndarray, pnp.ndarray, Sequence]):
        assert super_circ_params.shape[0] == self.p
        assert super_circ_params.shape[1] == self.c
        assert super_circ_params.shape[2] == self.l
        extracted_params = []
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        extracted_params = np.array(extracted_params)
        return 1-self.costFunc(extracted_params)

    def getGradient(self, super_circ_params:Union[np.ndarray, pnp.ndarray, Sequence]):
        assert super_circ_params.shape[0] == self.p
        assert super_circ_params.shape[1] == self.c
        assert super_circ_params.shape[2] == self.l
        extracted_params = []
        gradients = np.zeros(super_circ_params.shape)
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])

        if len(extracted_params) == 0:
            return gradients
        #cost_grad = jax.grad(self.costFunc, argnums=0)
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

class PrepareLogicalState513QECC(ModelFromK):
    coeff_dict = {
        'ket0':{'alpha':1, 'beta':0},
        'ket1':{'alpha':0, 'beta':1},
        'ket_plus':{'alpha':STATE_DIC['|+>'][0], 'beta':STATE_DIC['|+>'][1]},
        'ket_minus':{'alpha':STATE_DIC['|->'][0], 'beta':STATE_DIC['|->'][1]},
        'ket_plus_i':{'alpha':STATE_DIC['|+i>'][0], 'beta':STATE_DIC['|+i>'][1]},
        'ket_minus_i':{'alpha':STATE_DIC['|-i>'][0], 'beta':STATE_DIC['|-i>'][1]},
        'ket_T_state':{'alpha':STATE_DIC['magic'][0], 'beta':STATE_DIC['magic'][1]}
    }
    target_state_name = 'ket_plus'
    name = "PrepareLogicalState513QECC_Logical_"+target_state_name
    def __init__(self, p:int, c:int, l:int, structure_list:List[int], op_pool:Union[QMLPool, dict]):
        self.k = structure_list
        self.pool = op_pool
        self.p, self.c, self.l = p, c, l
        self.num_qubits = 5
        self.param_indices = extractParamIndicesQML(self.k, self.pool)
        self.dev = qml.device('default.qubit', wires = self.num_qubits)
        self.state_alpha = self.coeff_dict[self.target_state_name]['alpha']
        self.state_beta = self.coeff_dict[self.target_state_name]['beta']
        self.logical_X = np.kron(np.kron(np.kron(PAULI_X,np.kron(PAULI_X,PAULI_X)),PAULI_X),PAULI_X)
        self.logical_Y = np.kron(np.kron(np.kron(PAULI_Y,np.kron(PAULI_Y,PAULI_Y)),PAULI_Y),PAULI_Y)
        self.logical_Z = np.kron(np.kron(np.kron(PAULI_Z,np.kron(PAULI_Z,PAULI_Z)),PAULI_Z),PAULI_Z)
        self.O = (self.state_alpha*np.conj(self.state_beta)+np.conj(self.state_alpha)*self.state_beta)*self.logical_X-1j*(np.conj(self.state_alpha)*self.state_beta-self.state_alpha*np.conj(self.state_beta))*self.logical_Y-(self.state_beta*np.conj(self.state_beta)-self.state_alpha*np.conj(self.state_alpha))*self.logical_Z
        self.g1 = np.kron(np.kron(np.kron(np.kron(PAULI_X, PAULI_Z), PAULI_Z), PAULI_X), IDENTITY)
        self.g2 = np.kron(np.kron(np.kron(np.kron(IDENTITY, PAULI_X),PAULI_Z),PAULI_Z),PAULI_X)
        self.g3 = np.kron(np.kron(np.kron(np.kron(PAULI_X, IDENTITY), PAULI_X), PAULI_Z), PAULI_Z)
        self.g4 = np.kron(np.kron(np.kron(np.kron(PAULI_Z, PAULI_X), IDENTITY), PAULI_X), PAULI_Z)


        self.observable = qml.Hermitian(-1/5*(self.g1+self.g2+self.g3+self.g4+self.O), wires=[0,1,2,3,4])

    @qml.template
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
        @qml.qnode(self.dev)
        def fullCirc(extracted_params):
            self.backboneCirc(extracted_params)
            return qml.expval(self.observable)
        return fullCirc

    def costFunc(self, extracted_params):
        circ_func = self.constructFullCirc()
        energy = circ_func(extracted_params)
        return energy

    def getLoss(self, super_circ_params):
        extracted_params = []
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        extracted_params = np.array(extracted_params)
        cost = self.costFunc(extracted_params)
        return cost

    def getReward(self, super_circ_params):
        return -self.getLoss(super_circ_params)

    def getGradient(self, super_circ_params:Union[np.ndarray, pnp.ndarray, Sequence]):
        assert super_circ_params.shape[0] == self.p
        assert super_circ_params.shape[1] == self.c
        assert super_circ_params.shape[2] == self.l
        extracted_params = []
        gradients = np.zeros(super_circ_params.shape)
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])

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





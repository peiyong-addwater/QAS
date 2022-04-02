from .models import ModelFromK
from .qml_gate_ops import (
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
import qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import state_fidelity, DensityMatrix, Statevector
from qiskit.providers.aer.backends import StatevectorSimulator, AerSimulator, QasmSimulator
import qiskit.providers.aer.noise as noise
import shutup
shutup.please()


ket0 = np.array([1, 0])
ket1 = np.array([0, 1])
PAULI_EIGENSTATES_T_STATE = [(ket0 + ket1) / np.sqrt(2), (ket0 - ket1) / np.sqrt(2), ket0, ket1,
                             (ket0 + ket1 * 1j) / np.sqrt(2), (ket0 - ket1 * 1j) / np.sqrt(2),
                             (ket0 + np.exp(np.pi * 1j / 4) * ket1) / np.sqrt(2)]
THREE_QUBIT_BASIS = {
    '000':np.kron(ket0, np.kron(ket0, ket0)),
    '001':np.kron(ket0, np.kron(ket0, ket1)),
    '010':np.kron(ket0, np.kron(ket1, ket0)),
    '011':np.kron(ket0, np.kron(ket1, ket1)),
    '100':np.kron(ket1, np.kron(ket0, ket0)),
    '101':np.kron(ket1, np.kron(ket0, ket1)),
    '110':np.kron(ket1, np.kron(ket1, ket0)),
    '111':np.kron(ket1, np.kron(ket1, ket1))
}

PAULI_X = np.array([[0,1],[1,0]])
PAULI_Z = np.array(([[1,0],[0,-1]]))
PAULI_Y = np.array([[0,-1j],[1j,0]])
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
FULL_TOFFOLI_INPUT = []
_de = 1/np.sqrt(8) #0.354
# (qubit 0,1,2:) STZ
EXTENDED_TOFFILI_INPUT.append(np.array([ _de+0j, 0+_de*1j, 0.25+0.25j, -0.25+0.25j, -_de+0j, 0-_de*1j, -0.25-0.25j, 0.25-0.25j ]))
# T^dg, S^dg, Sqrt(X)
EXTENDED_TOFFILI_INPUT.append(np.array([ _de+0j, 0.25-0.25j, 0-_de*1j, -0.25-0.25j, _de+0j, 0.25-0.25j, 0-_de*1j, -0.25-0.25j ]))
# Y, Z, Sqrt(X)
EXTENDED_TOFFILI_INPUT.append(np.array([ 0-_de*1j, 0+_de*1j, 0+_de*1j, 0-_de*1j, 0-_de*1j, 0+_de*1j, 0+_de*1j, 0-_de*1j ]))
# P, Z, Y
EXTENDED_TOFFILI_INPUT.append(np.array([ 0-_de*1j, _de+0j, 0+_de*1j, -_de+0j, 0+_de*1j, -_de+0j, 0-_de*1j, _de+0j ]))

# \ket{000}+\ket{001}+\ket{010}+\ket{011}+\ket{100}+\ket{101}+\ket{110}-\ket{111}
EXTENDED_TOFFILI_INPUT.append(1/np.sqrt(8)*(THREE_QUBIT_BASIS['000']+THREE_QUBIT_BASIS['001']+
                                            THREE_QUBIT_BASIS['010']+THREE_QUBIT_BASIS['011']+
                                            THREE_QUBIT_BASIS['100']+THREE_QUBIT_BASIS['101']+
                                            THREE_QUBIT_BASIS['110']-THREE_QUBIT_BASIS['111']))

FULL_TOFFOLI_INPUT.extend(EXTENDED_TOFFILI_INPUT)
for a in [ket0, ket1]:
    for b in [ket0, ket1]:
        for c in [ket0, ket1]:
            temp = np.kron(a, b)
            s = np.kron(temp, c)
            TOFFOLI_INPUT.append(s)
for a in [ket0, ket1,(ket0 + np.exp(np.pi * 1j / 4) * ket1) / np.sqrt(2)]:
    for b in [ket0, ket1,(ket0 + np.exp(np.pi * 1j / 4) * ket1) / np.sqrt(2)]:
        for c in [ket0, ket1,(ket0 + np.exp(np.pi * 1j / 4) * ket1) / np.sqrt(2)]:
            temp = np.kron(a, b)
            s = np.kron(temp, c)
            EXTENDED_TOFFILI_INPUT.append(s)

for a in PAULI_EIGENSTATES_T_STATE:
    for b in PAULI_EIGENSTATES_T_STATE:
        for c in PAULI_EIGENSTATES_T_STATE:
            temp = np.kron(a, b)
            s = np.kron(temp, c)
            FULL_TOFFOLI_INPUT.append(s)
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
        FULL_TOFFOLI_INPUT.append(s)
        FULL_TOFFOLI_INPUT.append(s1)

THREE_QUBIT_GHZ_STATES = []
THREE_QUBIT_GHZ_STATES.append(1/np.sqrt(2)*(np.kron(ket0, np.kron(ket0, ket0))+np.kron(ket1, np.kron(ket1, ket1))))
THREE_QUBIT_GHZ_STATES.append(1/np.sqrt(2)*(np.kron(ket0, np.kron(ket0, ket0))-np.kron(ket1, np.kron(ket1, ket1))))
THREE_QUBIT_GHZ_STATES.append(1/np.sqrt(2)*(np.kron(ket0, np.kron(ket0, ket0))+1j*np.kron(ket1, np.kron(ket1, ket1))))
THREE_QUBIT_GHZ_STATES.append(1/np.sqrt(2)*(np.kron(ket0, np.kron(ket0, ket0))-1j*np.kron(ket1, np.kron(ket1, ket1))))
EXTENDED_TOFFILI_INPUT.extend(THREE_QUBIT_GHZ_STATES)
FOUR_TWO_TWO_DETECTION_CODE_INPUT.extend(TWO_QUBIT_ENTANGLED_STATES)
FULL_TOFFOLI_INPUT.extend(THREE_QUBIT_GHZ_STATES)

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

EXTENDED_TOFFOLI_DATA = []
for x in EXTENDED_TOFFILI_INPUT:
    dev_t = qml.device('default.qubit', wires=3)
    @qml.qnode(dev_t)
    def toffoli_circ(x):
        qml.QubitStateVector(x, wires=[0, 1, 2])
        qml.Toffoli(wires=[0, 1, 2])
        return qml.density_matrix(wires=[0, 1, 2])
    EXTENDED_TOFFOLI_DATA.append((x, toffoli_circ(x)))

FULL_TOFFOLI_DATA = []
for x in FULL_TOFFOLI_INPUT:
    dev_t = qml.device('default.qubit', wires=3)
    @qml.qnode(dev_t)
    def toffoli_circ(x):
        qml.QubitStateVector(x, wires=[0, 1, 2])
        qml.Toffoli(wires=[0, 1, 2])
        return qml.density_matrix(wires=[0, 1, 2])
    FULL_TOFFOLI_DATA.append((x, toffoli_circ(x)))

FOUR_TWO_TWO_DETECTION_CODE_DATA = []
for x in FOUR_TWO_TWO_DETECTION_CODE_INPUT:
    dev_422 = qml.device('default.qubit', wires=4)
    @qml.qnode(dev_422)
    def four_two_two_circ(x):
        qml.QubitStateVector(x, wires=[0, 1])
        qml.Hadamard(wires=[3])
        qml.CNOT(wires=[0,2])
        qml.CNOT(wires=[1,2])
        qml.CNOT(wires=[3,2])
        qml.CNOT(wires=[3,1])
        qml.CNOT(wires=[3,0])
        return qml.density_matrix(wires=[0,1,2,3])
    FOUR_TWO_TWO_DETECTION_CODE_DATA.append((x, four_two_two_circ(x)))

FIVE_ONE_THREE_QECC_DATA = []
for x in [ket0, ket1]:
    dev_513 = qml.device('default.qubit', wires = 5)
    @qml.qnode(dev_513)
    def five_one_three_circ(x):
        qml.QubitStateVector(x, wires=[0])
        qml.PauliZ(wires=[0])
        qml.Hadamard(wires=[2])
        qml.Hadamard(wires=[3])
        qml.adjoint(qml.S)(wires=[0])
        qml.CNOT(wires=[2,4])
        qml.CNOT(wires=[3,1])
        qml.Hadamard(wires=[1])
        qml.CNOT(wires=[3,4])
        qml.CNOT(wires=[1,0])
        qml.adjoint(qml.S)(wires=[2])
        qml.S(wires=[3])
        qml.adjoint(qml.S)(wires=4)
        qml.S(wires=0)
        qml.S(wires=1)
        qml.PauliZ(wires=2)
        qml.CNOT(wires=[4,0])
        qml.Hadamard(wires=4)
        qml.CNOT(wires=[4,1])
        return qml.density_matrix(wires=[0,1,2,3,4])
    FIVE_ONE_THREE_QECC_DATA.append((x, five_one_three_circ(x)))

TOFFOLI_012_MATRIX = qml.transforms.get_unitary_matrix(qml.Toffoli)(wires=[0,1,2])

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

    #@qml.template
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

    #@qml.template
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

class ToffoliQMLNoiselessAdditionalData(ModelFromK):
    name = "ToffoliQMLNoiselessAdditionalData"
    def __init__(self, p:int, c:int, l:int, structure_list:List[int], op_pool:Union[QMLPool, dict]):
        self.k = structure_list
        self.pool = op_pool
        self.p, self.c, self.l = p, c, l
        self.num_qubits = 3
        self.param_indices = extractParamIndicesQML(self.k, self.pool)
        self.data = EXTENDED_TOFFOLI_DATA
        self.dev = qml.device('default.qubit', wires = self.num_qubits)
        #self.dev = qml.device('qiskit.aer', wires=self.num_qubits, max_parallel_threads=0, max_parallel_experiments=0)

    #@qml.template
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

class ToffoliQMLNoiselessFullInput(ModelFromK):
    name = "ToffoliQMLNoiselessFullInput"
    def __init__(self, p:int, c:int, l:int, structure_list:List[int], op_pool:Union[QMLPool, dict]):
        self.k = structure_list
        self.pool = op_pool
        self.p, self.c, self.l = p, c, l
        self.num_qubits = 3
        self.param_indices = extractParamIndicesQML(self.k, self.pool)
        self.data = FULL_TOFFOLI_DATA
        self.dev = qml.device('default.qubit', wires = self.num_qubits)
        #self.dev = qml.device('qiskit.aer', wires=self.num_qubits, max_parallel_threads=0, max_parallel_experiments=0)

    #@qml.template
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

class ToffoliQMLSwapTestNoiselessExtendedData(ModelFromK):
    name = "ToffoliQMLSwapTestNoiselessExtendedData"
    def __init__(self, p:int, c:int, l:int, structure_list:List[int], op_pool:Union[QMLPool, dict]):
        self.k = structure_list
        self.pool = op_pool
        self.p, self.c, self.l = p, c, l
        self.num_qubits = 3
        self.param_indices = extractParamIndicesQML(self.k, self.pool)
        self.x_list = EXTENDED_TOFFILI_INPUT
        self.dev = qml.device('default.qubit', wires = self.num_qubits*2+1)
        #self.dev = qml.device('qiskit.aer', wires=self.num_qubits, max_parallel_threads=0, max_parallel_experiments=0)

    #@qml.template
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

class ToffoliQMLNoiselessUnitary(ModelFromK):
    name = "ToffoliQMLNoiselessUnitary"
    def __init__(self, p:int, c:int, l:int, structure_list:List[int], op_pool:Union[QMLPool, dict]):
        self.k = structure_list
        self.pool = op_pool
        self.p, self.c, self.l = p, c, l
        self.num_qubits = 3
        self.param_indices = extractParamIndicesQML(self.k, self.pool)
        self.data = EXTENDED_TOFFOLI_DATA
        self.dev = qml.device('default.qubit', wires = self.num_qubits)
        #self.dev = qml.device('qiskit.aer', wires=self.num_qubits, max_parallel_threads=0, max_parallel_experiments=0)

    #@qml.template
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
        def fullCirc(extracted_params):
            #qml.QubitStateVector(x, wires=[0,1,2])
            self.backboneCirc(extracted_params)
            #return qml.expval(qml.Hermitian(y, wires=[0,1,2]))
        return fullCirc

    def costFunc(self, extracted_params):
        circ_func = self.constructFullCirc()
        matrix_func = qml.transforms.get_unitary_matrix(circ_func,wire_order=[0, 1, 2])
        process_matrix = matrix_func(extracted_params)
        real_part = np.real(process_matrix)
        imag_part = np.imag(process_matrix)
        r_diff = np.linalg.norm(TOFFOLI_012_MATRIX-real_part)
        imag_diff = np.linalg.norm(imag_part)


        diff = r_diff+imag_diff

        return diff

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
        return -self.costFunc(extracted_params)

    def getGradient(self, super_circ_params:Union[np.ndarray, pnp.ndarray, Sequence]):
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
        self.data = FOUR_TWO_TWO_DETECTION_CODE_DATA
        self.dev = qml.device('default.qubit', wires = self.num_qubits)

    #@qml.template
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
        num_data = len(self.data)
        for i in range(num_data):
            fid = fid + circ_func(extracted_params, x=self.data[i][0], y=self.data[i][1])
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
        extracted_params = np.array(extracted_params, requires_grad=True)  # needed for the new pennylane version
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

class PrepareLogicalKetZeroState513QECC(ModelFromK):
    name = "PrepareLogicalState513QECC_LogicalKetZero"
    def __init__(self, p:int, c:int, l:int, structure_list:List[int], op_pool:Union[QMLPool, dict]):
        self.k = structure_list
        self.pool = op_pool
        self.p, self.c, self.l = p, c, l
        self.num_qubits = 5
        self.param_indices = extractParamIndicesQML(self.k, self.pool)
        self.dev = qml.device('default.qubit', wires = self.num_qubits)
        self.logical_0_vector = np.array([1/4, 0, 0, 1/4, 0, -(1/4), 1/4, 0, 0, -(1/4), -(1/4), 0, 1/4, 0, 0, -(1/4), 0, 1/4, -(1/4), 0, -(1/4), 0, 0, -(1/4), 1/4, 0, 0, -(1/4), 0, -(1/4), -(1/4), 0])# https://github.com/bernwo/five-qubit-code/blob/95ff95a3933baaa95ea3649cc28a2f1da4175af3/main.py#L20
        self.logical_1_vector = np.array([0, -(1/4), -(1/4), 0, -(1/4), 0, 0, 1/4, -(1/4), 0, 0, -(1/4), 0, -(1/4), 1/4, 0, -(1/4), 0, 0, 1/4, 0, -(1/4), -(1/4), 0, 0, 1/4, -(1/4), 0, 1/4, 0, 0, 1/4])# https://github.com/bernwo/five-qubit-code/blob/95ff95a3933baaa95ea3649cc28a2f1da4175af3/main.py#L32
        self.logical_0_observable = qml.Hermitian(np.outer(self.logical_0_vector, self.logical_0_vector),wires=[0,1,2,3,4])
        #self.observable = qml.Hermitian(-1/5*(self.g1+self.g2+self.g3+self.g4+self.O), wires=[0,1,2,3,4])

    #@qml.template
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
            return qml.expval(self.logical_0_observable)
        return fullCirc

    def costFunc(self, extracted_params):
        circ_func = self.constructFullCirc()
        overlap = circ_func(extracted_params)
        return 1-overlap

    def getLoss(self, super_circ_params):
        extracted_params = []
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        extracted_params = np.array(extracted_params)
        cost = self.costFunc(extracted_params)
        return cost

    def getReward(self, super_circ_params):
        return 1-self.getLoss(super_circ_params)

    def getGradient(self, super_circ_params:Union[np.ndarray, pnp.ndarray, Sequence]):
        assert super_circ_params.shape[0] == self.p
        assert super_circ_params.shape[1] == self.c
        assert super_circ_params.shape[2] == self.l
        extracted_params = []
        gradients = np.zeros(super_circ_params.shape)
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        extracted_params = np.array(extracted_params, requires_grad=True)  # needed for the new pennylane version
        #print(len(extracted_params))
        if len(extracted_params) == 0:
            return gradients
        cost_grad = qml.grad(self.costFunc)
        #extracted_params = np.array(extracted_params, requires_grad=True) # needed for the new pennylane version
        extracted_gradients = cost_grad(extracted_params)
        #print(extracted_params)
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

class PrepareLogicalKetMinusState513QECC(ModelFromK):
    coeff_dict = {
        'ket0':{'alpha':1, 'beta':0},
        'ket1':{'alpha':0, 'beta':1},
        'ket_plus':{'alpha':STATE_DIC['|+>'][0], 'beta':STATE_DIC['|+>'][1]},
        'ket_minus':{'alpha':STATE_DIC['|->'][0], 'beta':STATE_DIC['|->'][1]},
        'ket_plus_i':{'alpha':STATE_DIC['|+i>'][0], 'beta':STATE_DIC['|+i>'][1]},
        'ket_minus_i':{'alpha':STATE_DIC['|-i>'][0], 'beta':STATE_DIC['|-i>'][1]},
        'ket_T_state':{'alpha':STATE_DIC['magic'][0], 'beta':STATE_DIC['magic'][1]}
    }
    target_state_name = 'ket_minus'
    name = "PrepareLogicalKetMinusState513QECC"
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

    #@qml.template
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

class FiveOneThreeQECCNoiseless(ModelFromK):
    name = 'FiveOneThreeQECCNoiseless'
    def __init__(self, p:int, c:int, l:int, structure_list:List[int], op_pool:Union[QMLPool, dict]):
        self.k = structure_list
        self.pool = op_pool
        self.p, self.c, self.l = p, c, l
        self.num_qubits = 5
        self.param_indices = extractParamIndicesQML(self.k, self.pool)
        self.data = FIVE_ONE_THREE_QECC_DATA
        self.dev = qml.device('default.qubit', wires=5)

    #@qml.template
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
            qml.QubitStateVector(x, wires=[0])
            self.backboneCirc(extracted_params)
            return qml.expval(qml.Hermitian(y, wires=[0,1,2,3,4]))
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

"""
class TwoQubitH2(ModelFromK):
    name = 'TwoQubitH2'
    def __init__(self, p:int, c:int, l:int, structure_list:List[int], op_pool:Union[QMLPool, dict]):
        self.k = structure_list
        self.pool = op_pool
        self.p, self.c, self.l = p, c, l
        self.num_qubits = 2
        self.param_indices = extractParamIndicesQML(self.k, self.pool)
        self.dev = qml.device('default.qubit', wires=self.num_qubits)
        self.distance_and_coeffs = np.array([
    [0.05,  1.00777E+01,	-1.05533E+00,	1.55708E-01,	-1.05533E+00,	1.39333E-02],
    [0.10,	4.75665E+00,	-1.02731E+00,	1.56170E-01,	-1.02731E+00,	1.38667E-02],
    [0.15,	2.94817E+00,	-9.84234E-01,	1.56930E-01,	-9.84234E-01,	1.37610E-02],
    [0.20,	2.01153E+00,	-9.30489E-01,	1.57973E-01,	-9.30489E-01,	1.36238E-02],
    [0.25,	1.42283E+00,	-8.70646E-01,	1.59277E-01,	-8.70646E-01,	1.34635E-02],
    [0.30,	1.01018E+00,	-8.08649E-01,	1.60818E-01,	-8.08649E-01,	1.32880E-02],
    [0.35,	7.01273E-01,	-7.47416E-01,	1.62573E-01,	-7.47416E-01,	1.31036E-02],
    [0.40,	4.60364E-01,	-6.88819E-01,	1.64515E-01,	-6.88819E-01,	1.29140E-02],
    [0.45,	2.67547E-01,	-6.33890E-01,	1.66621E-01,	-6.33890E-01,	1.27192E-02],
    [0.50,	1.10647E-01,	-5.83080E-01,	1.68870E-01,	-5.83080E-01,	1.25165E-02],
    [0.55,	-1.83734E-02,	-5.36489E-01,	1.71244E-01,	-5.36489E-01,	1.23003E-02],
    [0.65,	-2.13932E-01,	-4.55433E-01,	1.76318E-01,	-4.55433E-01,	1.18019E-02],
    [0.75,	-3.49833E-01,	-3.88748E-01,	1.81771E-01,	-3.88748E-01,	1.11772E-02],
    [0.85,	-4.45424E-01,	-3.33747E-01,	1.87562E-01,    -3.33747E-01,	1.04061E-02],
    [0.95,	-5.13548E-01,	-2.87796E-01,	1.93650E-01,	-2.87796E-01,	9.50345E-03],
    [1.05,	-5.62600E-01,	-2.48783E-01,	1.99984E-01,	-2.48783E-01,	8.50998E-03],
    [1.15,	-5.97973E-01,	-2.15234E-01,	2.06495E-01,	-2.15234E-01,	7.47722E-03],
    [1.25,	-6.23223E-01,	-1.86173E-01,	2.13102E-01,	-1.86173E-01,	6.45563E-03],
    [1.35,	-6.40837E-01,	-1.60926E-01,	2.19727E-01,	-1.60926E-01,	5.48623E-03],
    [1.45,	-6.52661E-01,	-1.38977E-01,	2.26294E-01,	-1.38977E-01,	4.59760E-03],
    [1.55,	-6.60117E-01,	-1.19894E-01,	2.32740E-01,	-1.19894E-01,	3.80558E-03],
    [1.65,	-6.64309E-01,	-1.03305E-01,	2.39014E-01,	-1.03305E-01,	3.11545E-03],
    [1.75,	-6.66092E-01,	-8.88906E-02,	2.45075E-01,	-8.88906E-02,	2.52480E-03],
    [1.85,	-6.66126E-01,	-7.63712E-02,	2.50896E-01,	-7.63712E-02,	2.02647E-03],
    [1.95,	-6.64916E-01,	-6.55065E-02,	2.56458E-01,	-6.55065E-02,	1.61100E-03],
    [2.05,	-6.62844E-01,	-5.60866E-02,	2.61750E-01,	-5.60866E-02,	1.26812E-03],
    [2.15,	-6.60199E-01,	-4.79275E-02,	2.66768E-01,	-4.79275E-02,	9.88000E-04],
    [2.25,	-6.57196E-01,	-4.08672E-02,	2.71512E-01,	-4.08672E-02,	7.61425E-04],
    [2.35,	-6.53992E-01,	-3.47636E-02,	2.75986E-01,	-3.47636E-02,	5.80225E-04],
    [2.45,	-6.50702E-01,	-2.94924E-02,	2.80199E-01,	-2.94924E-02,	4.36875E-04],
    [2.55,	-6.47408E-01,	-2.49459E-02,	2.84160E-01,	-2.49459E-02,	3.25025E-04],
    [2.65,	-6.44165E-01,	-2.10309E-02,	2.87881E-01,	-2.10309E-02,	2.38800E-04],
    [2.75,	-6.41011E-01,	-1.76672E-02,	2.91376E-01,	-1.76672E-02,	1.73300E-04],
    [2.85,	-6.37971E-01,	-1.47853E-02,	2.94658E-01,	-1.47853E-02,	1.24200E-04],
    [2.95,	-6.35058E-01,	-1.23246E-02,	2.97741E-01,	-1.23246E-02,	8.78750E-05],
    [3.05,	-6.32279E-01,	-1.02318E-02,	3.00638E-01,	-1.02317E-02,	6.14500E-05],
    [3.15,	-6.29635E-01,	-8.45958E-03,	3.03362E-01,	-8.45958E-03,	4.24250E-05],
    [3.25,	-6.27126E-01,	-6.96585E-03,	3.05927E-01,	-6.96585E-03,	2.89500E-05],
    [3.35,	-6.24746E-01,	-5.71280E-03,	3.08344E-01,	-5.71280E-03,	1.95500E-05],
    [3.45,	-6.22491E-01,	-4.66670E-03,	3.10625E-01,	-4.66670E-03,	1.30500E-05],
    [3.55,	-6.20353E-01,	-3.79743E-03,	3.12780E-01,	-3.79743E-03,	8.57500E-06],
    [3.65,	-6.18325E-01,	-3.07840E-03,	3.14819E-01,	-3.07840E-03,	5.60000E-06],
    [3.75,	-6.16401E-01,	-2.48625E-03,	3.16750E-01,	-2.48625E-03,	3.60000E-06],
    [3.85,	-6.14575E-01,	-2.00063E-03,	3.18581E-01,	-2.00062E-03,	2.27500E-06],
    [3.95,	-6.12839E-01,	-1.60393E-03,	3.20320E-01,	-1.60392E-03,	1.42500E-06]
])
        self.ham = self.hamiltonianH2(*self.distance_and_coeffs[12][1:]) # 0.75 Angs, Min Energy = -1.1331360031249997 Ha

    def hamiltonianH2(self, coeff_I, coeff_Z1, coeff_X1X2, coeffZ_2, coeff_Z1Z2):
        return coeff_I * qml.Identity(0)@qml.Identity(1) + coeff_Z1 * qml.Identity(0) @ qml.PauliZ(1) + coeffZ_2 * qml.PauliZ(0) @ qml.Identity(1) + coeff_X1X2 * qml.PauliX(0) @ qml.PauliX(1) + coeff_Z1Z2 * qml.PauliZ(0) @ qml.PauliZ(1)

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
            return qml.expval(self.ham)
        return fullCirc

    def costFunc(self, extracted_params):
        circ_func = self.constructFullCirc()
        return circ_func(extracted_params)

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
        return -self.costFunc(extracted_params)

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

class TwoQubitH2Noisy(ModelFromK):
    name = 'TwoQubitH2Noisy'
    def __init__(self, p:int, c:int, l:int, structure_list:List[int], op_pool:Union[QMLPool, dict]):
        self.k = structure_list
        self.pool = op_pool
        self.p, self.c, self.l = p, c, l
        self.num_qubits = 2
        self.p1 = 0.001
        self.p2 = 0.01
        self.dp1 = noise.depolarizing_error(self.p1,1)
        self.dp2 = noise.depolarizing_error(self.p2,2)
        self.noise_model = noise.NoiseModel()
        self.noise_model.add_all_qubit_quantum_error(self.dp1, ['u1', 'u2', 'u3'])
        self.noise_model.add_all_qubit_quantum_error(self.dp2,['cx'])
        self.param_indices = extractParamIndicesQML(self.k, self.pool)
        self.dev = qml.device('qiskit.aer', wires=self.num_qubits, noise_model=self.noise_model)
        self.distance_and_coeffs = np.array([
    [0.05,  1.00777E+01,	-1.05533E+00,	1.55708E-01,	-1.05533E+00,	1.39333E-02],
    [0.10,	4.75665E+00,	-1.02731E+00,	1.56170E-01,	-1.02731E+00,	1.38667E-02],
    [0.15,	2.94817E+00,	-9.84234E-01,	1.56930E-01,	-9.84234E-01,	1.37610E-02],
    [0.20,	2.01153E+00,	-9.30489E-01,	1.57973E-01,	-9.30489E-01,	1.36238E-02],
    [0.25,	1.42283E+00,	-8.70646E-01,	1.59277E-01,	-8.70646E-01,	1.34635E-02],
    [0.30,	1.01018E+00,	-8.08649E-01,	1.60818E-01,	-8.08649E-01,	1.32880E-02],
    [0.35,	7.01273E-01,	-7.47416E-01,	1.62573E-01,	-7.47416E-01,	1.31036E-02],
    [0.40,	4.60364E-01,	-6.88819E-01,	1.64515E-01,	-6.88819E-01,	1.29140E-02],
    [0.45,	2.67547E-01,	-6.33890E-01,	1.66621E-01,	-6.33890E-01,	1.27192E-02],
    [0.50,	1.10647E-01,	-5.83080E-01,	1.68870E-01,	-5.83080E-01,	1.25165E-02],
    [0.55,	-1.83734E-02,	-5.36489E-01,	1.71244E-01,	-5.36489E-01,	1.23003E-02],
    [0.65,	-2.13932E-01,	-4.55433E-01,	1.76318E-01,	-4.55433E-01,	1.18019E-02],
    [0.75,	-3.49833E-01,	-3.88748E-01,	1.81771E-01,	-3.88748E-01,	1.11772E-02],
    [0.85,	-4.45424E-01,	-3.33747E-01,	1.87562E-01,    -3.33747E-01,	1.04061E-02],
    [0.95,	-5.13548E-01,	-2.87796E-01,	1.93650E-01,	-2.87796E-01,	9.50345E-03],
    [1.05,	-5.62600E-01,	-2.48783E-01,	1.99984E-01,	-2.48783E-01,	8.50998E-03],
    [1.15,	-5.97973E-01,	-2.15234E-01,	2.06495E-01,	-2.15234E-01,	7.47722E-03],
    [1.25,	-6.23223E-01,	-1.86173E-01,	2.13102E-01,	-1.86173E-01,	6.45563E-03],
    [1.35,	-6.40837E-01,	-1.60926E-01,	2.19727E-01,	-1.60926E-01,	5.48623E-03],
    [1.45,	-6.52661E-01,	-1.38977E-01,	2.26294E-01,	-1.38977E-01,	4.59760E-03],
    [1.55,	-6.60117E-01,	-1.19894E-01,	2.32740E-01,	-1.19894E-01,	3.80558E-03],
    [1.65,	-6.64309E-01,	-1.03305E-01,	2.39014E-01,	-1.03305E-01,	3.11545E-03],
    [1.75,	-6.66092E-01,	-8.88906E-02,	2.45075E-01,	-8.88906E-02,	2.52480E-03],
    [1.85,	-6.66126E-01,	-7.63712E-02,	2.50896E-01,	-7.63712E-02,	2.02647E-03],
    [1.95,	-6.64916E-01,	-6.55065E-02,	2.56458E-01,	-6.55065E-02,	1.61100E-03],
    [2.05,	-6.62844E-01,	-5.60866E-02,	2.61750E-01,	-5.60866E-02,	1.26812E-03],
    [2.15,	-6.60199E-01,	-4.79275E-02,	2.66768E-01,	-4.79275E-02,	9.88000E-04],
    [2.25,	-6.57196E-01,	-4.08672E-02,	2.71512E-01,	-4.08672E-02,	7.61425E-04],
    [2.35,	-6.53992E-01,	-3.47636E-02,	2.75986E-01,	-3.47636E-02,	5.80225E-04],
    [2.45,	-6.50702E-01,	-2.94924E-02,	2.80199E-01,	-2.94924E-02,	4.36875E-04],
    [2.55,	-6.47408E-01,	-2.49459E-02,	2.84160E-01,	-2.49459E-02,	3.25025E-04],
    [2.65,	-6.44165E-01,	-2.10309E-02,	2.87881E-01,	-2.10309E-02,	2.38800E-04],
    [2.75,	-6.41011E-01,	-1.76672E-02,	2.91376E-01,	-1.76672E-02,	1.73300E-04],
    [2.85,	-6.37971E-01,	-1.47853E-02,	2.94658E-01,	-1.47853E-02,	1.24200E-04],
    [2.95,	-6.35058E-01,	-1.23246E-02,	2.97741E-01,	-1.23246E-02,	8.78750E-05],
    [3.05,	-6.32279E-01,	-1.02318E-02,	3.00638E-01,	-1.02317E-02,	6.14500E-05],
    [3.15,	-6.29635E-01,	-8.45958E-03,	3.03362E-01,	-8.45958E-03,	4.24250E-05],
    [3.25,	-6.27126E-01,	-6.96585E-03,	3.05927E-01,	-6.96585E-03,	2.89500E-05],
    [3.35,	-6.24746E-01,	-5.71280E-03,	3.08344E-01,	-5.71280E-03,	1.95500E-05],
    [3.45,	-6.22491E-01,	-4.66670E-03,	3.10625E-01,	-4.66670E-03,	1.30500E-05],
    [3.55,	-6.20353E-01,	-3.79743E-03,	3.12780E-01,	-3.79743E-03,	8.57500E-06],
    [3.65,	-6.18325E-01,	-3.07840E-03,	3.14819E-01,	-3.07840E-03,	5.60000E-06],
    [3.75,	-6.16401E-01,	-2.48625E-03,	3.16750E-01,	-2.48625E-03,	3.60000E-06],
    [3.85,	-6.14575E-01,	-2.00063E-03,	3.18581E-01,	-2.00062E-03,	2.27500E-06],
    [3.95,	-6.12839E-01,	-1.60393E-03,	3.20320E-01,	-1.60392E-03,	1.42500E-06]
])
        self.ham = self.hamiltonianH2(*self.distance_and_coeffs[12][1:]) # 0.75 Angs, Min Energy = -1.1331360031249997 Ha

    def hamiltonianH2(self, coeff_I, coeff_Z1, coeff_X1X2, coeffZ_2, coeff_Z1Z2):
        return coeff_I * qml.Identity(0)@qml.Identity(1) + coeff_Z1 * qml.Identity(0) @ qml.PauliZ(1) + coeffZ_2 * qml.PauliZ(0) @ qml.Identity(1) + coeff_X1X2 * qml.PauliX(0) @ qml.PauliX(1) + coeff_Z1Z2 * qml.PauliZ(0) @ qml.PauliZ(1)

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
            return qml.expval(self.ham)
        return fullCirc

    def costFunc(self, extracted_params):
        circ_func = self.constructFullCirc()
        return circ_func(extracted_params)

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
        return -self.costFunc(extracted_params)

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
"""

_H2_SYMBOLS = ["H", "H"]
_H2_COORDINATES = np.array([0.0, 0.0, -0.6614, 0.0, 0.0, 0.6614])
_H2_HAM_FOUR_QUBIT, _ = qml.qchem.molecular_hamiltonian(_H2_SYMBOLS, _H2_COORDINATES) #ground-state energy = -1.13618883 Ha

_LiH_SYMBOLS = ["Li", "H"]
_LiH_COORDINATES = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 2.969280527]) # units in Bohr
_LiH_HAM, _LiH_QUBITS = qml.qchem.molecular_hamiltonian(_LiH_SYMBOLS, _LiH_COORDINATES) # ground-state energy = -7.8825378193 Ha


class FourQubitH2(ModelFromK):

    name = "FourQubitH2"

    def __init__(self, p: int, c: int, l: int, structure_list: List[int], op_pool: Union[QMLPool, dict]):
        self.k = structure_list
        self.pool = op_pool
        self.p, self.c, self.l = p, c, l
        self.num_qubits = 4
        self.num_electrons = 2
        self.param_indices = extractParamIndicesQML(self.k, self.pool)
        self.dev = qml.device('default.qubit', wires=self.num_qubits)
        self.H= _H2_HAM_FOUR_QUBIT  # ground-state energy = -1.13618883 Ha
        self.hf = qml.qchem.hf_state(self.num_electrons, self.num_qubits)

    #@qml.template
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
            qml.BasisState(self.hf, wires=[0,1,2,3])
            self.backboneCirc(extracted_params)
            return qml.expval(self.H)
        return fullCirc

    def costFunc(self, extracted_params):
        circ_func = self.constructFullCirc()
        return circ_func(extracted_params)

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
        return -self.costFunc(extracted_params)

    def getGradient(self, super_circ_params:Union[np.ndarray, pnp.ndarray, Sequence]):
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


"""
class FourQubitH2Noisy(ModelFromK):

    name = "FourQubitH2Noisy"

    def __init__(self, p: int, c: int, l: int, structure_list: List[int], op_pool: Union[QMLPool, dict]):
        self.k = structure_list
        self.pool = op_pool
        self.p, self.c, self.l = p, c, l
        self.num_qubits = 4
        self.num_electrons = 2
        self.param_indices = extractParamIndicesQML(self.k, self.pool)
        self.p1 = 0.001
        self.p2 = 0.01

        self.param_indices = extractParamIndicesQML(self.k, self.pool)
        self.dev = qml.device('default.mixed', wires=self.num_qubits)
        self.H= _H2_HAM_FOUR_QUBIT  # ground-state energy = -1.13618883 Ha
        self.hf = qml.qchem.hf_state(self.num_electrons, self.num_qubits)

    #@qml.template
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
            qml.BasisState(self.hf, wires=[0,1,2,3])
            self.backboneCirc(extracted_params)
            for i in range(self.num_qubits):
                qml.DepolarizingChannel(self.p1, wires=i)
            return qml.expval(self.H)
        return fullCirc

    def costFunc(self, extracted_params):
        circ_func = self.constructFullCirc()
        return circ_func(extracted_params)

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
        return -self.costFunc(extracted_params)

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

class FourQubitH2CleanStart(ModelFromK):

    name = "FourQubitH2CleanStart"

    def __init__(self, p: int, c: int, l: int, structure_list: List[int], op_pool: Union[QMLPool, dict]):
        self.k = structure_list
        self.pool = op_pool
        self.p, self.c, self.l = p, c, l
        self.num_qubits = 4
        self.num_electrons = 2
        self.param_indices = extractParamIndicesQML(self.k, self.pool)
        self.dev = qml.device('default.qubit', wires=self.num_qubits)
        self.H= _H2_HAM_FOUR_QUBIT  # ground-state energy = -1.13618883 Ha
        self.hf = qml.qchem.hf_state(self.num_electrons, self.num_qubits)

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
            #qml.BasisState(self.hf, wires=[0,1,2,3])
            self.backboneCirc(extracted_params)
            return qml.expval(self.H)
        return fullCirc

    def costFunc(self, extracted_params):
        circ_func = self.constructFullCirc()
        return circ_func(extracted_params)

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
        return -self.costFunc(extracted_params)

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
"""


class LiH(ModelFromK):

    name = "LiH"

    def __init__(self, p: int, c: int, l: int, structure_list: List[int], op_pool: Union[QMLPool, dict]):
        self.k = structure_list
        self.pool = op_pool
        self.p, self.c, self.l = p, c, l
        self.num_qubits = _LiH_QUBITS
        self.num_active_electrons = 2
        self.param_indices = extractParamIndicesQML(self.k, self.pool)
        self.dev = qml.device('default.qubit', wires=self.num_qubits)
        self.H= qml.utils.sparse_hamiltonian(_LiH_HAM)  # ground-state energy = -7.8825378193 Ha
        self.hf = qml.qchem.hf_state(self.num_active_electrons, self.num_qubits)

    #@qml.template
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
        @qml.qnode(self.dev,diff_method="parameter-shift")
        def fullCirc(extracted_params):
            qml.BasisState(self.hf, wires=range(self.num_qubits))
            self.backboneCirc(extracted_params)
            return qml.expval(qml.SparseHamiltonian(self.H, wires=range(self.num_qubits)))
        return fullCirc

    def costFunc(self, extracted_params):
        circ_func = self.constructFullCirc()
        return circ_func(extracted_params)

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
        return -self.costFunc(extracted_params)

    def getGradient(self, super_circ_params:Union[np.ndarray, pnp.ndarray, Sequence]):
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

class WStateFiveQubit(ModelFromK):
    name = 'WStateFiveQubit'
    def __init__(self, p:int, c:int, l:int, structure_list:List[int], op_pool:Union[QMLPool, dict]):
        self.k = structure_list
        self.pool = op_pool
        self.p, self.c, self.l = p, c, l
        self.num_qubits = 5
        self.param_indices = extractParamIndicesQML(self.k, self.pool)
        self.dev = qml.device('default.qubit', wires=self.num_qubits)
        self.target_state = 1/np.sqrt(5)*(np.kron(ket1, np.kron(ket0, np.kron(ket0, np.kron(ket0, ket0))))+
                                          np.kron(ket0, np.kron(ket1, np.kron(ket0, np.kron(ket0, ket0)))) +
                                          np.kron(ket0, np.kron(ket0, np.kron(ket1, np.kron(ket0, ket0)))) +
                                          np.kron(ket0, np.kron(ket0, np.kron(ket0, np.kron(ket1, ket0)))) +
                                          np.kron(ket0, np.kron(ket0, np.kron(ket0, np.kron(ket0, ket1))))
                                          )
        self.observable = qml.Hermitian(np.outer(self.target_state, self.target_state), wires = [0,1,2,3,4])

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
        overlap = circ_func(extracted_params)
        return 1-overlap

    def getLoss(self, super_circ_params):
        extracted_params = []
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        extracted_params = np.array(extracted_params)
        cost = self.costFunc(extracted_params)
        return cost

    def getReward(self, super_circ_params):
        return 1-self.getLoss(super_circ_params)

    def getGradient(self, super_circ_params:Union[np.ndarray, pnp.ndarray, Sequence]):
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


class VQLSDemo(ModelFromK):
    name = 'VQLSDemo'
    def __init__(self, p:int, c:int, l:int, structure_list:List[int], op_pool:Union[QMLPool, dict]):
        self.k = structure_list
        self.pool = op_pool
        self.p, self.c, self.l = p, c, l
        self.num_qubits = 4
        self.param_indices = extractParamIndicesQML(self.k, self.pool)
        self.dev = qml.device('default.qubit', wires=self.num_qubits)

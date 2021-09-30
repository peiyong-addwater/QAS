from .models import ModelFromK
from .qml_ops import (
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
from qiskit.quantum_info import state_fidelity, DensityMatrix
from qiskit.providers.aer.backends import StatevectorSimulator, AerSimulator, QasmSimulator

ket0 = np.array([1, 0])
ket1 = np.array([0, 1])
PAULI_EIGENSTATES_T_STATE = [(ket0 + ket1) / np.sqrt(2), (ket0 - ket1) / np.sqrt(2), ket0, ket1,
                             (ket0 + ket1 * 1j) / np.sqrt(2), (ket0 - ket1 * 1j) / np.sqrt(2),
                             (ket0 + np.exp(np.pi * 1j / 4) * ket1) / np.sqrt(2)]

STATE_DIC = {'|0>': ket0, '|1>': ket1, '|+>': (ket0 + ket1) / np.sqrt(2), '|->': (ket0 - ket1) / np.sqrt(2),
             '|+i>': (ket0 + ket1 * 1j) / np.sqrt(2), '|-i>': (ket0 - ket1 * 1j) / np.sqrt(2),
             'magic': (ket0 + np.exp(np.pi * 1j / 4) * ket1) / np.sqrt(2)}
STATE_DICT_KEYS = list(STATE_DIC.keys())

FOUR_TWO_TWO_DETECTION_CODE_INPUT = []
for c in PAULI_EIGENSTATES_T_STATE:
    for d in PAULI_EIGENSTATES_T_STATE:
        FOUR_TWO_TWO_DETECTION_CODE_INPUT.append(np.kron(c,d))

TOFFOLI_INPUT = []
for a in [ket0, ket1]:
    for b in [ket0, ket1]:
        for c in [ket0, ket1]:
            temp = np.kron(a, b)
            s = np.kron(temp, c)
            TOFFOLI_INPUT.append(s)

def extractParamIndicesQML(k:List[int], op_pool:QMLPool)->List:
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


def get_data_ccx_gate(init_states:List[np.ndarray])->List[DensityMatrix]:
    backbone_circ = QuantumCircuit(3)
    backbone_circ.ccx(0, 1, 2)
    encoded_states = []
    for state in init_states:
        qc = QuantumCircuit(3)
        qc.initialize(state, [0,1,2])
        qc.append(backbone_circ.to_instruction(), [0, 1, 2])
        qc.save_density_matrix(label='encoded_state')
        simulator = AerSimulator(max_parallel_threads=0, max_parallel_experiments=0)
        result = simulator.run(qiskit.transpile(qc, simulator)).result().data()[
            'encoded_state']
        encoded_states.append(DensityMatrix(result).data)
    return encoded_states

def generate_single_qubit_state(theta_B:float, phi_B:float)->np.ndarray:
    return np.cos(theta_B/2)*ket0 + np.sin(theta_B/2)*np.exp(1j*phi_B)*ket1

def get_encoded_states_ideal_bit_flip_code(init_states:List[np.ndarray])->List[DensityMatrix]:
    backbone_circ = QuantumCircuit(3)
    backbone_circ.cnot(control_qubit=0,target_qubit=1)
    backbone_circ.cnot(control_qubit=0, target_qubit=2)
    encoded_states = []
    for state in init_states:
        qc = QuantumCircuit(3)
        qc.initialize(state, 0)
        qc.append(backbone_circ.to_instruction(), [0,1,2])
        qc.save_density_matrix(label='encoded_state')
        simulator = AerSimulator(max_parallel_threads=0, max_parallel_experiments=0)
        result = simulator.run(qiskit.transpile(qc, simulator)).result().data()[
            'encoded_state']
        encoded_states.append(DensityMatrix(result).data)
    return encoded_states

def get_encoded_states_ideal_phase_flip_code(init_states:List[np.ndarray])->List[DensityMatrix]:
    backbone_circ = QuantumCircuit(3)
    backbone_circ.h(0)
    backbone_circ.cnot(control_qubit=0,target_qubit=1)
    backbone_circ.cnot(control_qubit=0, target_qubit=2)
    encoded_states = []
    for state in init_states:
        qc = QuantumCircuit(3)
        qc.initialize(state, 0)
        qc.append(backbone_circ.to_instruction(), [0, 1, 2])
        qc.save_density_matrix(label='encoded_state')
        simulator = AerSimulator(max_parallel_threads=0, max_parallel_experiments=0)
        result = simulator.run(qiskit.transpile(qc, simulator)).result().data()[
            'encoded_state']
        encoded_states.append(DensityMatrix(result).data)
    return encoded_states

def get_encoded_states_ideal_shor_9_bit_code(init_states:List[np.ndarray])->List[DensityMatrix]:
    phase_flip = QuantumCircuit(3)
    phase_flip.cnot(control_qubit=0, target_qubit=1)
    phase_flip.cnot(control_qubit=0, target_qubit=2)
    phase_flip.h(0)
    phase_flip.h(1)
    phase_flip.h(2)
    bit_flip = QuantumCircuit(3)
    bit_flip.cnot(0,1)
    bit_flip.cnot(0,2)

    encoded_states = []
    for state in init_states:
        qc = QuantumCircuit(9)
        qc.initialize(state, 0)
        qc.append(phase_flip.to_gate(), [0,3,6])
        qc.append(bit_flip.to_gate(), [0,1,2])
        qc.append(bit_flip.to_gate(), [3,4,5])
        qc.append(bit_flip.to_gate(), [6,7,8])
        qc.save_density_matrix(label='encoded_state')
        simulator = AerSimulator(max_parallel_threads=0, max_parallel_experiments=0)
        result = simulator.run(qiskit.transpile(qc, simulator)).result().data()[
            'encoded_state']
        encoded_states.append(DensityMatrix(result).data)
    return encoded_states

def get_encoded_states_ideal_five_bit_code(init_states:List[np.ndarray])->List[DensityMatrix]:
    backbone_circ = QuantumCircuit(5)
    backbone_circ.z(0)
    backbone_circ.h(2)
    backbone_circ.sdg(0)
    backbone_circ.cnot(2,4)
    backbone_circ.h(3)
    backbone_circ.cnot(3,1)
    backbone_circ.x(4)
    backbone_circ.h(1)
    backbone_circ.cu(0,0,-np.pi/2, 0, control_qubit=3, target_qubit=4) #csdg3,4
    backbone_circ.cnot(1,0)
    backbone_circ.s(3)
    backbone_circ.s(0)
    backbone_circ.s(1)
    backbone_circ.cnot(4,0)
    backbone_circ.h(4)
    backbone_circ.cnot(4,1)
    backbone_circ.sdg(2)
    backbone_circ.z(2)

    encoded_states = []
    for state in init_states:
        qc = QuantumCircuit(5)
        qc.initialize(state, 0)
        qc.append(backbone_circ.to_instruction(), [0, 1, 2, 3, 4])
        qc.save_density_matrix(label='encoded_state')
        simulator = AerSimulator(max_parallel_threads=0, max_parallel_experiments=0)
        result = simulator.run(qiskit.transpile(qc, simulator)).result().data()[
            'encoded_state']
        encoded_states.append(DensityMatrix(result).data)
    return encoded_states

def get_code_space_422_detection_code(init_states)->List[DensityMatrix]:
    backbone_circ = QuantumCircuit(4)
    backbone_circ.cnot(0,2)
    backbone_circ.h(3)
    backbone_circ.cnot(1,2)
    backbone_circ.cnot(3,2)
    backbone_circ.cnot(3,1)
    backbone_circ.cnot(3,0)
    encoded_states = []
    for state in init_states:
        qc = QuantumCircuit(4)
        qc.initialize(state, [0,1])
        qc.append(backbone_circ.to_instruction(), [0,1,2,3])
        qc.save_density_matrix(label='encoded_state')
        simulator = AerSimulator(max_parallel_threads=0, max_parallel_experiments=0)
        result = simulator.run(qiskit.transpile(qc, simulator)).result().data()[
            'encoded_state']
        encoded_states.append(DensityMatrix(result).data)
    return encoded_states

SIMPLE_DATASET_BIT_FLIP = []
SIMPLE_DATASET_BIT_FLIP.append(PAULI_EIGENSTATES_T_STATE)
SIMPLE_DATASET_BIT_FLIP.append(get_encoded_states_ideal_bit_flip_code(PAULI_EIGENSTATES_T_STATE))

SIMPLE_DATASET_PHASE_FLIP = []
SIMPLE_DATASET_PHASE_FLIP.append(PAULI_EIGENSTATES_T_STATE)
SIMPLE_DATASET_PHASE_FLIP.append(get_encoded_states_ideal_phase_flip_code(PAULI_EIGENSTATES_T_STATE))

SIMPLE_DATASET_FIVE_BIT_CODE = []
SIMPLE_DATASET_FIVE_BIT_CODE.append(PAULI_EIGENSTATES_T_STATE)
SIMPLE_DATASET_FIVE_BIT_CODE.append(get_encoded_states_ideal_five_bit_code(PAULI_EIGENSTATES_T_STATE))

SIMPLE_DATASET_NINE_BIT_CODE = []
SIMPLE_DATASET_NINE_BIT_CODE.append(PAULI_EIGENSTATES_T_STATE)
SIMPLE_DATASET_NINE_BIT_CODE.append(get_encoded_states_ideal_shor_9_bit_code(PAULI_EIGENSTATES_T_STATE))

FOUR_TWO_TWO_DETECTION_CODE_DATA = []
FOUR_TWO_TWO_DETECTION_CODE_DATA.append(FOUR_TWO_TWO_DETECTION_CODE_INPUT)
FOUR_TWO_TWO_DETECTION_CODE_DATA.append(get_code_space_422_detection_code(FOUR_TWO_TWO_DETECTION_CODE_INPUT))

TOFFOLI_DATA = []
TOFFOLI_DATA.append(TOFFOLI_INPUT)
TOFFOLI_DATA.append(get_data_ccx_gate(TOFFOLI_INPUT))

class ToffoliQMLNoiseless(ModelFromK):
    def __init__(self, p:int, c:int, l:int, structure_list:List[int], op_pool:QMLPool):
        self.k = structure_list
        self.pool = op_pool
        self.p, self.c, self.l = p, c, l
        self.num_qubits = 3
        self.param_indices = extractParamIndicesQML(self.k, self.pool)
        self.x_list = TOFFOLI_DATA[0]
        self.y_list = TOFFOLI_DATA[1]
        #self.dev = qml.device('qulacs.simulator', wires=self.num_qubits, gpu=True)
        self.dev = qml.device('default.qubit', wires = self.num_qubits)

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
        num_data = len(self.x_list)
        for i in range(num_data):
            fid = fid + circ_func(extracted_params, x=self.x_list[i], y=self.y_list[i])
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

class PhaseFlipQMLNoiseless(ModelFromK):
    def __init__(self, p:int, c:int, l:int, structure_list:List[int], op_pool:QMLPool):
        self.k = structure_list
        self.pool = op_pool
        self.p, self.c, self.l = p, c, l
        self.num_qubits = 3
        self.param_indices = extractParamIndicesQML(self.k, self.pool)
        self.x_list = SIMPLE_DATASET_PHASE_FLIP[0]
        self.y_list = SIMPLE_DATASET_PHASE_FLIP[1]
        self.dev = qml.device('default.qubit', wires = self.num_qubits)

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
            qml.QubitStateVector(x, wires=[0])
            self.backboneCirc(extracted_params)
            return qml.expval(qml.Hermitian(y, wires=[0,1,2]))
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
        extracted_params = pnp.array(extracted_params)
        return self.costFunc(extracted_params)

    def getReward(self, super_circ_params:Union[np.ndarray, pnp.ndarray, Sequence]):
        assert super_circ_params.shape[0] == self.p
        assert super_circ_params.shape[1] == self.c
        assert super_circ_params.shape[2] == self.l
        extracted_params = []
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])
        extracted_params = pnp.array(extracted_params)
        return 1-self.costFunc(extracted_params)

    def getGradient(self, super_circ_params:Union[np.ndarray, pnp.ndarray, Sequence]):
        assert super_circ_params.shape[0] == self.p
        assert super_circ_params.shape[1] == self.c
        assert super_circ_params.shape[2] == self.l
        extracted_params = []
        for index in self.param_indices:
            extracted_params.append(super_circ_params[index])

        gradients = np.zeros(super_circ_params.shape)
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


class FourTwoTwoNoiseless(ModelFromK):
    def __init__(self, p:int, c:int, l:int, structure_list:List[int], op_pool:QMLPool):
        self.k = structure_list
        self.pool = op_pool
        self.p, self.c, self.l = p, c, l
        self.num_qubits = 4
        self.param_indices = extractParamIndicesQML(self.k, self.pool)
        self.x_list = FOUR_TWO_TWO_DETECTION_CODE_DATA[0]
        self.y_list = FOUR_TWO_TWO_DETECTION_CODE_DATA[1]
        #self.dev = qml.device('qulacs.simulator', wires=self.num_qubits, gpu=True)
        self.dev = qml.device('default.qubit', wires = self.num_qubits)

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








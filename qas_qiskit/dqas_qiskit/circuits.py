from standard_ops import *
import qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import state_fidelity, DensityMatrix
import numpy as np
from qiskit.providers.aer.backends import StatevectorSimulator, AerSimulator, QasmSimulator
# Import from Qiskit Aer noise module
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise import QuantumError, ReadoutError
from qiskit.providers.aer.noise import pauli_error
from qiskit.providers.aer.noise import depolarizing_error
from qiskit.providers.aer.noise import thermal_relaxation_error
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Unroller

ket0 = np.array([1, 0])
ket1 = np.array([0, 1])
PAULI_EIGENSTATES_T_STATE = [(ket0 + ket1) / np.sqrt(2), (ket0 - ket1) / np.sqrt(2), ket0, ket1,
                             (ket0 + ket1 * 1j) / np.sqrt(2), (ket0 - ket1 * 1j) / np.sqrt(2),
                             (ket0 + np.exp(np.pi * 1j / 4) * ket1) / np.sqrt(2)]

STATE_DIC = {'|0>': ket0, '|1>': ket1, '|+>': (ket0 + ket1) / np.sqrt(2), '|->': (ket0 - ket1) / np.sqrt(2),
             '|+i>': (ket0 + ket1 * 1j) / np.sqrt(2), '|-i>': (ket0 - ket1 * 1j) / np.sqrt(2),
             'magic': (ket0 + np.exp(np.pi * 1j / 4) * ket1) / np.sqrt(2)}
STATE_DICT_KEYS = list(STATE_DIC.keys())


def generate_single_qubit_state(theta_B:float, phi_B:float)->np.ndarray:
    return np.cos(theta_B/2)*ket0 + np.sin(theta_B/2)*np.exp(1j*phi_B)*ket1

def extract_ops(num_qubits:int, k:List[int],op_pool:GatePool, circ_params:np.ndarray)->(List[QuantumGate], List):
    p, c, l = circ_params.shape[0], circ_params.shape[1], circ_params.shape[2]
    assert num_qubits == op_pool.num_qubits
    assert p == len(k)
    assert c == len(op_pool)
    assert l == 3 # force max num of parameters in each gate equal to 3
    assert min(k) >=0
    assert max(k) <c
    extracted_gates = []
    param_indices = []
    for i in range(len(k)):
        gate_info =op_pool[k[i]]
        assert len(gate_info.keys()) == 1
        gate_name = list(gate_info.keys())[0]
        gate_pos = gate_info[gate_name]
        #TODO: make modifications to the code for single- and two-parameter gates
        gate_param = circ_params[i, k[i], :]
        param_indices.append([[i, k[i], 0], [i, k[i], 1], [i, k[i], 2]])
        gate = QuantumGate(gate_name, gate_pos, gate_param)
        extracted_gates.append(gate)
    return extracted_gates, param_indices

def construct_backbone_circuit_from_gate_list(num_qubits:int, extracted_gates:List[QuantumGate])->QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    for c in extracted_gates:
        qc.append(c.get_op(), qargs=c.get_qreg_pos())
    return qc

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
        encoded_states.append(DensityMatrix(result))
    return encoded_states

def get_encoded_states_ideal_phase_flip_code(init_states:List[np.ndarray])->List[DensityMatrix]:
    backbone_circ = QuantumCircuit(3)
    backbone_circ.cnot(control_qubit=0,target_qubit=1)
    backbone_circ.cnot(control_qubit=0, target_qubit=2)
    backbone_circ.h(0)
    backbone_circ.h(1)
    backbone_circ.h(2)
    encoded_states = []
    for state in init_states:
        qc = QuantumCircuit(3)
        qc.initialize(state, 0)
        qc.append(backbone_circ.to_instruction(), [0, 1, 2])
        qc.save_density_matrix(label='encoded_state')
        simulator = AerSimulator(max_parallel_threads=0, max_parallel_experiments=0)
        result = simulator.run(qiskit.transpile(qc, simulator)).result().data()[
            'encoded_state']
        encoded_states.append(DensityMatrix(result))
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
        encoded_states.append(DensityMatrix(result))
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
        encoded_states.append(DensityMatrix(result))
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


class QCircFromK(ABC):

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def get_loss(self):
        pass

    @abstractmethod
    def get_gradient(self):
        pass


    @abstractmethod
    def get_extracted_QuantumCircuit_object(self):
        pass


class BitFlipSearchDensityMatrix(QCircFromK):
    def __init__(self, circ_params:np.ndarray, structure_list:List[int], op_pool:GatePool):
        self.params = circ_params
        self.k = structure_list
        self.pool = op_pool
        self.p, self.c, self.l = circ_params.shape[0], circ_params.shape[1], circ_params.shape[2]
        self.extracted_gates, self.param_indices = extract_ops(3, self.k, self.pool, self.params)
        self.backbone_circ = construct_backbone_circuit_from_gate_list(3, self.extracted_gates)
        self.init_states = SIMPLE_DATASET_BIT_FLIP[0]
        self.target_states = SIMPLE_DATASET_BIT_FLIP[1]
        self.loss = self.calculate_avg_loss_with_prepend_states(self.init_states, self.target_states,
                                                                self.backbone_circ)
        self.gradient = self.parameter_shift_gradient()

    def parameter_shift_gradient(self):
        shift = np.pi/2
        gradients = np.zeros((self.p, self.c, self.l))
        for gate_param_indices in self.param_indices:
            for index in gate_param_indices:
                shift_mat = np.zeros((self.p, self.c, self.l))
                shift_mat[index[0], index[1], index[2]] = shift
                right_shifted_params = self.params + shift_mat
                left_shifted_params = self.params-shift_mat
                right_shifted_gates, _ = extract_ops(3, self.k, self.pool, right_shifted_params)
                left_shifted_gates, _ = extract_ops(3, self.k, self.pool, left_shifted_params)
                right_shifted_circ = construct_backbone_circuit_from_gate_list(3, right_shifted_gates)
                left_shifted_circ = construct_backbone_circuit_from_gate_list(3, left_shifted_gates)
                right_shifted_loss = self.calculate_avg_loss_with_prepend_states(self.init_states, self.target_states,
                                                                                 right_shifted_circ)
                left_shifted_loss = self.calculate_avg_loss_with_prepend_states(self.init_states, self.target_states,
                                                                                left_shifted_circ)
                gradients[index[0], index[1], index[2]] = (right_shifted_loss-left_shifted_loss)/2
        return gradients


    def __str__(self):
        gate_name_list = [str(c) for c in self.extracted_gates]
        return gate_name_list

    def get_extracted_QuantumCircuit_object(self):
        return self.backbone_circ

    def calculate_avg_loss_with_prepend_states(self, init_states:List[np.ndarray],
                                               target_states:List[DensityMatrix],
                                               backbone_circ:QuantumCircuit,)->np.float64:
        num_qubits = backbone_circ.num_qubits
        simulator = AerSimulator(max_parallel_threads=0, max_parallel_experiments=0)
        fid_list = []
        for i in range(len(init_states)):
            target = target_states[i]
            qc = QuantumCircuit(num_qubits)
            qc.initialize(init_states[i], 0)
            qc.append(backbone_circ.to_instruction(), [i for i in range(num_qubits)])
            qc.save_density_matrix(label='encoded_state')
            qc = qiskit.transpile(qc, simulator)
            result = simulator.run(qc).result().data()['encoded_state']
            result = DensityMatrix(result)
            fidelity = state_fidelity(result, target)
            fid_list.append(fidelity)
        return 1-np.average(fid_list)

    def get_loss(self):
        return self.loss

    def get_gradient(self):
        return self.gradient



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

def extract_ops(num_qubits:int, k:List[int],op_pool:GatePool, circ_params:np.ndarray)->List[QuantumGate]:
    p, c, l = circ_params.shape[0], circ_params.shape[1], circ_params.shape[2]
    assert num_qubits == op_pool.num_qubits
    assert p == len(k)
    assert c == len(op_pool)
    assert l == 3 # force max num of parameters in each gate equal to 3
    assert min(k) >=0
    assert max(k) <c
    extracted_gates = []
    for i in range(len(k)):
        gate_info =op_pool[k[i]]
        assert len(gate_info.keys()) == 1
        gate_name = list(gate_info.keys())[0]
        gate_pos = gate_info[gate_name]
        gate_param = circ_params[i, k[i], :]
        gate = QuantumGate(gate_name, gate_pos, gate_param)
        extracted_gates.append(gate)
    return extracted_gates

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

simple_dataset_bit_flip = []
simple_dataset_bit_flip.append(PAULI_EIGENSTATES_T_STATE)
simple_dataset_bit_flip.append(get_encoded_states_ideal_bit_flip_code(PAULI_EIGENSTATES_T_STATE))

simple_dataset_phase_flip = []
simple_dataset_phase_flip.append(PAULI_EIGENSTATES_T_STATE)
simple_dataset_phase_flip.append(get_encoded_states_ideal_phase_flip_code(PAULI_EIGENSTATES_T_STATE))

simple_dataset_5_bit_code = []
simple_dataset_5_bit_code.append(PAULI_EIGENSTATES_T_STATE)
simple_dataset_5_bit_code.append(get_encoded_states_ideal_five_bit_code(PAULI_EIGENSTATES_T_STATE))

simple_dataset_9_bit_code = []
simple_dataset_9_bit_code.append(PAULI_EIGENSTATES_T_STATE)
simple_dataset_9_bit_code.append(get_encoded_states_ideal_shor_9_bit_code(PAULI_EIGENSTATES_T_STATE))


class QCircFromk(ABC):

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def get_full_circuit_with_prepend(self):
        pass

    @abstractmethod
    def get_loss(self):
        pass

    @abstractmethod
    def get_gradient(self):
        pass

    @abstractmethod
    def get_final_state_density_matrix(self):
        pass

    @abstractmethod
    def get_measurement_results(self):
        pass

    @abstractmethod
    def get_extracted_QuantumCircuit_object(self):
        pass



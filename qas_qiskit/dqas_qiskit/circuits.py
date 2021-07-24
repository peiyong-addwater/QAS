from .standard_ops import *
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
import jax.numpy as jnp
import optax
from joblib import delayed, Parallel

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
for c in [ket0, ket1]:
    for d in [ket0, ket1]:
        FOUR_TWO_TWO_DETECTION_CODE_INPUT.append(np.kron(c, d))


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

def get_code_space_422_detection_code(init_states:List[np.ndarray])->List[DensityMatrix]:
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

FOUR_TWO_TWO_DETECTION_CODE_DATA = []
FOUR_TWO_TWO_DETECTION_CODE_DATA.append(FOUR_TWO_TWO_DETECTION_CODE_INPUT)
FOUR_TWO_TWO_DETECTION_CODE_DATA.append(get_code_space_422_detection_code(FOUR_TWO_TWO_DETECTION_CODE_INPUT))


class QCircFromK(ABC):

    @abstractmethod
    def __init__(self, *args):
        pass

    @abstractmethod
    def get_loss(self, *args):
        pass

    @abstractmethod
    def get_gradient(self, *args):
        pass


    @abstractmethod
    def get_extracted_QuantumCircuit_object(self, *args):
        pass

    @abstractmethod
    def penalty_terms(self, *args):
        pass

class SearchDensityMatrix(QCircFromK):

    @abstractmethod
    def __init__(self, p:int, c:int, l:int, structure_list:List[int], op_pool:GatePool,
                 noise_model:Optional[NoiseModel]=None):
        self.k = structure_list
        self.pool = op_pool
        self.p, self.c, self.l = p, c, l
        self.noise_model = noise_model

    @abstractmethod
    def get_loss(self, *args):
        pass

    @abstractmethod
    def get_gradient(self, *args):
        pass

    @abstractmethod
    def get_extracted_QuantumCircuit_object(self, *args):
        pass

    def gradient_parameter_shift_single_parameter(self, num_qubits:int, circ_params:np.ndarray,
                                                  index:List[int],init_states:List, target_states:List, shift=np.pi/2):
        gradient = np.zeros((self.p, self.c, self.l))
        shift_mat = np.zeros((self.p, self.c, self.l))
        shift_mat[index[0], index[1], index[2]] = shift
        right_shifted_params = circ_params + shift_mat
        left_shifted_params = circ_params - shift_mat
        right_shifted_gates, _ = extract_ops(num_qubits, self.k, self.pool, right_shifted_params)
        left_shifted_gates, _ = extract_ops(num_qubits, self.k, self.pool, left_shifted_params)
        right_shifted_circ = construct_backbone_circuit_from_gate_list(num_qubits, right_shifted_gates)
        left_shifted_circ = construct_backbone_circuit_from_gate_list(num_qubits, left_shifted_gates)
        right_shifted_loss = self.calculate_avg_loss_with_prepend_states(init_states, target_states,
                                                                         right_shifted_circ)
        left_shifted_loss = self.calculate_avg_loss_with_prepend_states(init_states, target_states,
                                                                        left_shifted_circ)
        gradient[index[0], index[1], index[2]] = (right_shifted_loss-left_shifted_loss)/2

        return gradient

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
            try:
                fidelity = state_fidelity(result, target, validate=False)
            except:
                if not result.is_valid():
                    print("Invalid Result State Encountered, Setting Fidelity To Zero")
                fidelity = 0
                pass
            fid_list.append(fidelity)
        return 1-np.average(fid_list)

    def penalty_terms(self, *args):
        raise NotImplementedError

class FourTwoTwoDetectionDensityMatrixNoiseless(SearchDensityMatrix):
    def __init__(self, p:int, c:int, l:int, structure_list:List[int], op_pool:GatePool):
        self.k = structure_list
        self.pool = op_pool
        self.p, self.c, self.l = p,c,l
        self.num_qubits = 4

        # self.gradient = self.parameter_shift_gradient()

    def parameter_shift_gradient(self, circ_params, param_indices, init_states, target_states):
        shift = np.pi/2
        all_param_indices = []
        for gate_param_indices in param_indices:
            for index in gate_param_indices:
                all_param_indices.append(index)
        gradients = Parallel(n_jobs=-1, verbose=0)(delayed(self.gradient_parameter_shift_single_parameter)(
            self.num_qubits, circ_params, index, init_states, target_states, shift
        ) for index in all_param_indices)
        gradients = jnp.stack(gradients, axis=0)
        gradients = jnp.sum(gradients, axis=0)
        return gradients


    def get_extracted_QuantumCircuit_object(self, circ_params):
        extracted_gates, param_indices = extract_ops(self.num_qubits, self.k, self.pool, circ_params)
        backbone_circ = construct_backbone_circuit_from_gate_list(3, extracted_gates)
        return backbone_circ


    def get_loss(self, circ_params, init_states, target_states):
        assert self.p == circ_params.shape[0]
        assert self.c == circ_params.shape[1]
        assert self.l == circ_params.shape[2]
        extracted_gates, param_indices = extract_ops(self.num_qubits, self.k, self.pool, circ_params)
        backbone_circ = construct_backbone_circuit_from_gate_list(self.num_qubits, extracted_gates)
        loss = self.calculate_avg_loss_with_prepend_states(init_states, target_states, backbone_circ)

        return loss+self.penalty_terms(circ_params)

    def get_gradient(self, circ_params,init_states, target_states):
        assert self.p == circ_params.shape[0]
        assert self.c == circ_params.shape[1]
        assert self.l == circ_params.shape[2]
        _, param_indices = extract_ops(self.num_qubits, self.k, self.pool, circ_params)

        return self.parameter_shift_gradient(circ_params, param_indices, init_states, target_states)

    def get_circuit_ops(self, circ_params):
        extracted_gates, _ = extract_ops(self.num_qubits, self.k, self.pool, circ_params)
        op_list = [str(c) for c in extracted_gates]
        return op_list

    def penalty_terms(self, circ_params):
        # penalty for consecutive repeated operations
        repeated = 0
        for i in range(1, len(self.k)):
            if self.k[i-1] == self.k[i]:
                repeated = repeated + 1

        # penalty for "useless" gates
        qubits_with_action = set()
        qubits_with_action.add(0)
        num_useless = 0
        extracted_gates,_ = extract_ops(self.num_qubits, self.k, self.pool, circ_params)
        for op in extracted_gates:
            qubit_list = op.get_qreg_pos()
            if len(qubit_list) == 1:
                qubits_with_action.add(qubit_list[0])
            if len(qubit_list) == 2:
                if qubit_list[0] not in qubits_with_action:
                    num_useless = num_useless + 1
                else:
                    qubits_with_action.add(qubit_list[1])

        return (repeated + num_useless)/1



class PhaseFlipDensityMatrixNoiseless(SearchDensityMatrix):
    def __init__(self, p:int, c:int, l:int, structure_list:List[int], op_pool:GatePool):
        self.k = structure_list
        self.pool = op_pool
        self.p, self.c, self.l = p,c,l
        self.num_qubits = 3

        # self.gradient = self.parameter_shift_gradient()

    def parameter_shift_gradient(self, circ_params, param_indices, init_states, target_states):
        shift = np.pi/2
        all_param_indices = []
        for gate_param_indices in param_indices:
            for index in gate_param_indices:
                all_param_indices.append(index)
        gradients = Parallel(n_jobs=-1, verbose=0)(delayed(self.gradient_parameter_shift_single_parameter)(
            self.num_qubits, circ_params, index, init_states, target_states, shift
        ) for index in all_param_indices)
        gradients = jnp.stack(gradients, axis=0)
        gradients = jnp.sum(gradients, axis=0)
        return gradients


    def get_extracted_QuantumCircuit_object(self, circ_params):
        extracted_gates, param_indices = extract_ops(self.num_qubits, self.k, self.pool, circ_params)
        backbone_circ = construct_backbone_circuit_from_gate_list(self.num_qubits, extracted_gates)
        return backbone_circ


    def get_loss(self, circ_params, init_states, target_states):
        assert self.p == circ_params.shape[0]
        assert self.c == circ_params.shape[1]
        assert self.l == circ_params.shape[2]
        extracted_gates, param_indices = extract_ops(self.num_qubits, self.k, self.pool, circ_params)
        backbone_circ = construct_backbone_circuit_from_gate_list(self.num_qubits, extracted_gates)
        loss = self.calculate_avg_loss_with_prepend_states(init_states, target_states, backbone_circ)

        return loss + self.penalty_terms(circ_params)

    def get_gradient(self, circ_params,init_states, target_states):
        assert self.p == circ_params.shape[0]
        assert self.c == circ_params.shape[1]
        assert self.l == circ_params.shape[2]
        _, param_indices = extract_ops(self.num_qubits, self.k, self.pool, circ_params)

        return self.parameter_shift_gradient(circ_params, param_indices, init_states, target_states)

    def get_circuit_ops(self, circ_params):
        extracted_gates, _ = extract_ops(self.num_qubits, self.k, self.pool, circ_params)
        op_list = [str(c) for c in extracted_gates]
        return op_list

    def penalty_terms(self, circ_params):
        # penalty for consecutive repeated operations
        repeated = 0
        for i in range(1, len(self.k)):
            if self.k[i-1] == self.k[i]:
                repeated = repeated + 1

        # penalty for "useless" gates
        qubits_with_action = set()
        qubits_with_action.add(0)
        num_useless = 0
        extracted_gates,_ = extract_ops(self.num_qubits, self.k, self.pool, circ_params)
        for op in extracted_gates:
            qubit_list = op.get_qreg_pos()
            if len(qubit_list) == 1:
                qubits_with_action.add(qubit_list[0])
            if len(qubit_list) == 2:
                if qubit_list[0] not in qubits_with_action:
                    num_useless = num_useless + 1
                else:
                    qubits_with_action.add(qubit_list[1])

        return (repeated + num_useless)/1


class BitFlipSearchDensityMatrixNoiseless(SearchDensityMatrix):
    def __init__(self, p:int, c:int, l:int, structure_list:List[int], op_pool:GatePool):
        self.k = structure_list
        self.pool = op_pool
        self.p, self.c, self.l = p,c,l
        self.num_qubits = 3

        # self.gradient = self.parameter_shift_gradient()

    def parameter_shift_gradient(self, circ_params, param_indices, init_states, target_states):
        shift = np.pi/2
        all_param_indices = []
        for gate_param_indices in param_indices:
            for index in gate_param_indices:
                all_param_indices.append(index)
        gradients = Parallel(n_jobs=-1, verbose=0)(delayed(self.gradient_parameter_shift_single_parameter)(
            self.num_qubits, circ_params, index, init_states, target_states, shift
        ) for index in all_param_indices)
        gradients = jnp.stack(gradients, axis=0)
        gradients = jnp.sum(gradients, axis=0)
        return gradients


    def get_extracted_QuantumCircuit_object(self, circ_params):
        extracted_gates, param_indices = extract_ops(self.num_qubits, self.k, self.pool, circ_params)
        backbone_circ = construct_backbone_circuit_from_gate_list(self.num_qubits, extracted_gates)
        return backbone_circ


    def get_loss(self, circ_params, init_states, target_states):
        assert self.p == circ_params.shape[0]
        assert self.c == circ_params.shape[1]
        assert self.l == circ_params.shape[2]
        extracted_gates, param_indices = extract_ops(self.num_qubits, self.k, self.pool, circ_params)
        backbone_circ = construct_backbone_circuit_from_gate_list(self.num_qubits, extracted_gates)
        loss = self.calculate_avg_loss_with_prepend_states(init_states, target_states, backbone_circ)

        return loss + self.penalty_terms(circ_params)

    def get_gradient(self, circ_params,init_states, target_states):
        assert self.p == circ_params.shape[0]
        assert self.c == circ_params.shape[1]
        assert self.l == circ_params.shape[2]
        _, param_indices = extract_ops(self.num_qubits, self.k, self.pool, circ_params)

        return self.parameter_shift_gradient(circ_params, param_indices, init_states, target_states)

    def get_circuit_ops(self, circ_params):
        extracted_gates, _ = extract_ops(self.num_qubits, self.k, self.pool, circ_params)
        op_list = [str(c) for c in extracted_gates]
        return op_list

    def penalty_terms(self, circ_params):
        # penalty for consecutive repeated operations
        repeated = 0
        for i in range(1, len(self.k)):
            if self.k[i-1] == self.k[i]:
                repeated = repeated + 1

        # penalty for "useless" gates
        qubits_with_action = set()
        qubits_with_action.add(0)
        num_useless = 0
        extracted_gates,_ = extract_ops(self.num_qubits, self.k, self.pool, circ_params)
        for op in extracted_gates:
            qubit_list = op.get_qreg_pos()
            if len(qubit_list) == 1:
                qubits_with_action.add(qubit_list[0])
            if len(qubit_list) == 2:
                if qubit_list[0] not in qubits_with_action:
                    num_useless = num_useless + 1
                else:
                    qubits_with_action.add(qubit_list[1])

        return (repeated + num_useless)/1

class FiveBitCodeSearchDensityMatrixNoiseless(SearchDensityMatrix):
    def __init__(self, p:int, c:int, l:int, structure_list:List[int], op_pool:GatePool):
        self.k = structure_list
        self.pool = op_pool
        self.p, self.c, self.l = p,c,l
        self.num_qubits = 5

        # self.gradient = self.parameter_shift_gradient()

    def parameter_shift_gradient(self, circ_params, param_indices, init_states, target_states):
        shift = np.pi / 2
        all_param_indices = []
        for gate_param_indices in param_indices:
            for index in gate_param_indices:
                all_param_indices.append(index)
        gradients = Parallel(n_jobs=-1, verbose=0)(delayed(self.gradient_parameter_shift_single_parameter)(
            self.num_qubits, circ_params, index, init_states, target_states, shift
        ) for index in all_param_indices)
        gradients = jnp.stack(gradients, axis=0)
        gradients = jnp.sum(gradients, axis=0)
        return gradients


    def get_extracted_QuantumCircuit_object(self, circ_params):
        extracted_gates, param_indices = extract_ops(self.num_qubits, self.k, self.pool, circ_params)
        backbone_circ = construct_backbone_circuit_from_gate_list(5, extracted_gates)
        return backbone_circ

    def get_loss(self, circ_params, init_states, target_states):
        assert self.p == circ_params.shape[0]
        assert self.c == circ_params.shape[1]
        assert self.l == circ_params.shape[2]
        extracted_gates, param_indices = extract_ops(self.num_qubits, self.k, self.pool, circ_params)
        backbone_circ = construct_backbone_circuit_from_gate_list(self.num_qubits, extracted_gates)
        loss = self.calculate_avg_loss_with_prepend_states(init_states, target_states, backbone_circ)

        return loss + self.penalty_terms(circ_params)

    def get_gradient(self, circ_params,init_states, target_states):
        assert self.p == circ_params.shape[0]
        assert self.c == circ_params.shape[1]
        assert self.l == circ_params.shape[2]
        _, param_indices = extract_ops(self.num_qubits, self.k, self.pool, circ_params)

        return self.parameter_shift_gradient(circ_params, param_indices, init_states, target_states)

    def get_circuit_ops(self, circ_params):
        extracted_gates, _ = extract_ops(self.num_qubits, self.k, self.pool, circ_params)
        op_list = [str(c) for c in extracted_gates]
        return op_list

    def penalty_terms(self, circ_params):
        # penalty for consecutive repeated operations
        repeated = 0
        for i in range(1, len(self.k)):
            if self.k[i-1] == self.k[i]:
                repeated = repeated + 1

        # penalty for "useless" gates
        qubits_with_action = set()
        qubits_with_action.add(0)
        num_useless = 0
        extracted_gates,_ = extract_ops(self.num_qubits, self.k, self.pool, circ_params)
        for op in extracted_gates:
            qubit_list = op.get_qreg_pos()
            if len(qubit_list) == 1:
                qubits_with_action.add(qubit_list[0])
            if len(qubit_list) == 2:
                if qubit_list[0] not in qubits_with_action:
                    num_useless = num_useless + 1
                else:
                    qubits_with_action.add(qubit_list[1])

        return (repeated + num_useless)/1

"""
pool = default_complete_graph_parameterized_pool(3)
p = 7
c = len(pool)
l = 3
print(pool)
k = [3,4,4,6,8,3,2]
params = np.random.rand(p*c*l).reshape((p,c,l))

start_learning_rate = 1e-1
optimizer = optax.adam(start_learning_rate)
opt_state = optimizer.init(params)
for i in range(50):
    print(i)
    circ = BitFlipSearchDensityMatrix(p,c,l, k, pool)
    print(circ.get_circuit_ops(params))
    grads = circ.get_gradient(params)
    grads = jnp.nan_to_num(grads)
    loss = circ.get_loss(params)
    print(loss)
    #print(grads)
    updates, opt_state = optimizer.update(grads, opt_state)
    #print(updates)
    params = optax.apply_updates(params, updates)
    params = np.array(params)
"""


from standard_ops import *
import qiskit
from qiskit import QuantumCircuit
import numpy as np

ket0 = np.array([1, 0])
ket1 = np.array([0, 1])
PAULI_EIGENSTATES_T_STATE = [(ket0 + ket1) / np.sqrt(2), (ket0 - ket1) / np.sqrt(2), ket0, ket1,
                             (ket0 + ket1 * 1j) / np.sqrt(2), (ket0 - ket1 * 1j) / np.sqrt(2),
                             (ket0 + np.exp(np.pi * 1j / 4) * ket1) / np.sqrt(2)]

STATE_DIC = {'|0>': ket0, '|1>': ket1, '|+>': (ket0 + ket1) / np.sqrt(2), '|->': (ket0 - ket1) / np.sqrt(2),
             '|+i>': (ket0 + ket1 * 1j) / np.sqrt(2), '|-i>': (ket0 - ket1 * 1j) / np.sqrt(2),
             'magic': (ket0 + np.exp(np.pi * 1j / 4) * ket1) / np.sqrt(2)}
STATE_DICT_KEYS = list(STATE_DIC.keys())

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

"""
k = [3,5,2]
pool = default_complete_graph_parameterized_pool(3)
params = np.random.randn(3*len(pool)*3).reshape((3, len(pool), 3))
egs = extract_ops(3, k, pool,params)
print([str(c) for c in egs])
"""

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
    def get_qasm(self):
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


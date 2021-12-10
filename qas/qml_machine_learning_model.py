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
import qiskit.providers.aer.noise as noise
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.decomposition import PCA
import shutup
shutup.please()

# The data "challenge.npz" is from https://github.com/quantum-melb/MCDS-workshop/blob/main/day-four/data/challenge.npz
# Load the Fashion-MNIST data in the folder "additional_data"
# original labels are 0 and 3
DATA_PATH = './additional_data/challenge.npz'
data_F_MNIST = np.load(DATA_PATH)
sample_train = data_F_MNIST['sample_train']
labels_train = data_F_MNIST['labels_train']

# Split train data
sample_train_F_MNIST, sample_val_F_MNIST, labels_train_F_MNIST, labels_val_F_MNIST = train_test_split(
    sample_train, labels_train, test_size=0.2, random_state=42)

# Load test data and change label 3 to 1 and 0 to -1
sample_test_F_MNIST = data_F_MNIST['sample_test']
labels_test_F_MNIST = data_F_MNIST['labels_test']
for i in range(len(labels_train_F_MNIST)):
    if labels_train_F_MNIST[i] == 3:
        labels_train_F_MNIST[i] = 1

    if labels_train_F_MNIST[i] == 0:
        labels_train_F_MNIST[i] = -1

for i in range(len(labels_val_F_MNIST)):
    if labels_val_F_MNIST[i] == 3:
        labels_val_F_MNIST[i] = 1

    if labels_val_F_MNIST[i] == 0:
        labels_val_F_MNIST[i] = -1

for i in range(len(labels_test_F_MNIST)):
    if labels_test_F_MNIST[i] == 3:
        labels_test_F_MNIST[i] = 1

    if labels_test_F_MNIST[i] == 0:
        labels_test_F_MNIST[i] = -1

# Data transformation from https://github.com/quantum-melb/MCDS-workshop/blob/main/day-four/classification_challenge.ipynb
# Standardize
standard_scaler = StandardScaler()
sample_train_F_MNIST = standard_scaler.fit_transform(sample_train_F_MNIST)
sample_val_F_MNIST = standard_scaler.transform(sample_val_F_MNIST)
sample_test_F_MNIST = standard_scaler.transform(sample_test_F_MNIST)

# Reduce dimensions
N_DIM = 16
pca = PCA(n_components=N_DIM)
sample_train_F_MNIST = pca.fit_transform(sample_train_F_MNIST)
sample_val_F_MNIST = pca.transform(sample_val_F_MNIST)
sample_test_F_MNIST = pca.transform(sample_test_F_MNIST)

# Normalize
min_max_scaler = MinMaxScaler((-1, 1))
sample_train_F_MNIST = min_max_scaler.fit_transform(sample_train_F_MNIST)
sample_val_F_MNIST = min_max_scaler.transform(sample_val_F_MNIST)
sample_test_F_MNIST = min_max_scaler.transform(sample_test_F_MNIST)

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

class BinaryClassificationFashionMNIST(ModelFromK):

    name = "BinaryClassificationFashionMNIST"

    def __init__(self, p: int, c: int, l: int, structure_list: List[int], op_pool: Union[QMLPool, dict]):
        self.k = structure_list
        self.pool = op_pool
        self.p, self.c, self.l = p, c, l
        self.num_qubits = 4
        self.num_electrons = 2
        self.param_indices = extractParamIndicesQML(self.k, self.pool)
        self.dev = qml.device('default.qubit', wires=self.num_qubits)
        self.train_data = pnp.array(sample_train_F_MNIST, requires_grad=False)
        self.train_label = pnp.array(labels_train_F_MNIST, requires_grad=False)
        self.val_data = pnp.array(sample_val_F_MNIST, requires_grad=False)
        self.val_label = pnp.array(labels_val_F_MNIST, requires_grad=False)

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
        def fullCirc(extracted_params, x=None):
            qml.AmplitudeEmbedding(features=x, wires=[0,1,2,3])
            self.backboneCirc(extracted_params)
            return qml.expval(qml.PauliZ(0))
        return fullCirc

    # https://pennylane.ai/qml/demos/tutorial_variational_classifier.html
    def square_loss(self, labels, predictions):
        loss = 0
        for l, p in zip(labels, predictions):
            loss = loss + (l - p) ** 2

        loss = loss / len(labels)
        return loss

    # https://pennylane.ai/qml/demos/tutorial_variational_classifier.html
    # This will be the reward
    def accuracy(self, labels, predictions):

        loss = 0
        for l, p in zip(labels, predictions):
            if abs(l - p) < 1e-5:
                loss = loss + 1
        loss = loss / len(labels)

        return loss

    def costFunc(self, extracted_params):
        circ_func = self.constructFullCirc()
        preds = []
        for x in self.train_data:
            preds.append(circ_func(extracted_params, x=x))

        loss = self.square_loss(self.train_label, preds)
        return loss

    def accFunc(self, extracted_params):
        circ_func = self.constructFullCirc()
        preds = []
        for x in self.train_data:
            preds.append(circ_func(extracted_params, x=x))

        loss = self.accuracy(self.train_label, preds)
        return loss

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
        return self.accFunc(extracted_params)

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







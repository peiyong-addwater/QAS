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
DATA_PATH = './additional_data/challenge.npz'
data = np.load(DATA_PATH)
sample_train = data['sample_train']
labels_train = data['labels_train']

# Split train data
sample_train, sample_val, labels_train, labels_val = train_test_split(
    sample_train, labels_train, test_size=0.2, random_state=42)

# Load test data
sample_test = data['sample_test']
labels_test = data['labels_test']


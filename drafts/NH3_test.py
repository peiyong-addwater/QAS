import pennylane as qml
from pennylane import numpy as np
from pennylane import qchem
import os

cwd = os.getcwd()
print(cwd)
symbols, coordinates = qchem.read_structure("NH3.xyz")
H, qubits = qchem.molecular_hamiltonian(symbols, coordinates)
print("Number of qubits: {:}".format(qubits))
print("Qubit Hamiltonian")
print(H)
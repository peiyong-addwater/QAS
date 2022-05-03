import pennylane as qml
from pennylane import numpy as np
from pennylane import qchem
import os

symbols, coordinates = qchem.read_structure("NH3.xyz")
H, qubits = qchem.molecular_hamiltonian(symbols, coordinates)
print("Number of qubits: {:}".format(qubits))
print("Qubit Hamiltonian")
print(H)
active_electrons = 8
active_orbitals = 8
singles, doubles = qchem.excitations(active_electrons, qubits)

print(f"Total number of excitations = {len(singles) + len(doubles)}")
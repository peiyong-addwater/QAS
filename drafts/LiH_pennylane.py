import pennylane as qml
from pennylane import qchem
from pennylane import numpy as np
import time

symbols = ["Li", "H"]
geometry = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 2.969280527])

H, qubits = qchem.molecular_hamiltonian(
    symbols,
    geometry,
    active_electrons=2,
    active_orbitals=5
)

active_electrons = 2

singles, doubles = qchem.excitations(active_electrons, qubits)

print(f"Single excitations (2 CNOTs, 1 CRY): {len(singles)}")
print(f"Double excitations (14 CNOTs, 8 RY, 6 Hadarmad): {len(doubles)}")
print(f"Total number of excitations = {len(singles) + len(doubles)}")
print(f"Total number of basic gates: {len(singles)*3 + len(doubles)*28}")
print(f"Total number of CNOTs(CRYs): {len(singles)*3+len(doubles)*14}")


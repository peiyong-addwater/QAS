import pennylane as qml
import pennylane.numpy as np
from pennylane.transforms.draw import draw
from qiskit.quantum_info import state_fidelity, DensityMatrix, Statevector
import inspect
import time
from qiskit import QuantumCircuit
dev = qml.device('qiskit.aer', wires=7)
"""
qc = QuantumCircuit(3)
qc.h(0)
qc.cx(0,1)
qc.cx(1,2)
dm = DensityMatrix(qc).data
print(dm)
"""

print(np.outer(np.array([1,0]), np.array([1,0])))
@qml.qnode(dev)
def circuit():
    for i in range(7):
        qml.Hadamard(wires=i)
    qml.Toffoli(wires=[0,1,2])
    qml.Toffoli(wires=[3,4,5])
    qml.CSWAP(wires=[6, 5, 2])
    qml.CSWAP(wires=[6, 4, 1])
    qml.CSWAP(wires=[6, 3, 0])
    qml.Hadamard(wires=6)
    return qml.expval(qml.Hermitian(np.outer(np.array([1,0]), np.array([1,0])),wires=6))

#circ = circuit()
#print(qml.draw(circuit)())
#print(circuit())
@qml.qnode(dev)
def circ2(theta):
    qml.RX(theta, wires=0)
    return qml.probs(wires=0)

angles = np.linspace(0, 2*np.pi, 10)
res = [circ2(theta) for theta in angles]
for a, t in zip(angles,res):
    print(a, t, np.argmax(t))

print(res[0])

res = np.array(res)
print(np.argmax(res, axis = -1))
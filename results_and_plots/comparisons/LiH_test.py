import pennylane.numpy as np
from pennylane import qchem
import pennylane as qml
import time
import pyscf
from pyscf import gto
symbols = ["Li", "H"]
geometry = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 2.969280527])

"""
converged SCF energy = -7.80324297491405
E(FCI) = -7.888567271126
converged SCF energy = -7.80324297491408  <S^2> = 5.5511151e-15  2S+1 = 1
E(UHF-FCI) = -7.888567271127
E(FCI) = -7.888567271126
"""

basis_set = "sto-6g"
mol = pyscf.M(
    atom = 'Li 0. 0. 0.; H 0.0 0.0 2.969280527',
    basis = basis_set,
    symmetry = True,
)
mol.unit = 'B'
myhf = mol.RHF().run()

#
# create an FCI solver based on the SCF object
#
cisolver = pyscf.fci.FCI(myhf)
print('E(FCI) = %.12f' % cisolver.kernel()[0])

#
# create an FCI solver based on the SCF object
#
myuhf = mol.UHF().run()
cisolver = pyscf.fci.FCI(myuhf)
print('E(UHF-FCI) = %.12f' % cisolver.kernel()[0])

#
# create an FCI solver based on the given orbitals and the num. electrons and
# spin of the mol object
#
cisolver = pyscf.fci.FCI(mol, myhf.mo_coeff)
print('E(FCI) = %.12f' % cisolver.kernel()[0])






H, qubits = qchem.molecular_hamiltonian(
    symbols,
    geometry,
    active_electrons=2,
    active_orbitals=5,
    basis=basis_set
)

active_electrons = 2

singles, doubles = qchem.excitations(active_electrons, qubits)

print(f"Total number of excitations = {len(singles) + len(doubles)}")

print("Number of qubits required to perform quantum simulations: {:}".format(qubits))




hf_state = qchem.hf_state(active_electrons, qubits)


def circuit_1(params, wires, excitations):
    qml.BasisState(hf_state, wires=wires)

    for i, excitation in enumerate(excitations):
        if len(excitation) == 4:
            qml.DoubleExcitation(params[i], wires=excitation)
        else:
            qml.SingleExcitation(params[i], wires=excitation)

dev = qml.device("default.qubit", wires=qubits)
cost_fn = qml.ExpvalCost(circuit_1, H, dev, optimize=True)

circuit_gradient = qml.grad(cost_fn, argnum=0)

params = [0.0] * len(doubles)
grads = circuit_gradient(params, excitations=doubles)

for i in range(len(doubles)):
    print(f"Excitation : {doubles[i]}, Gradient: {grads[i]}")

doubles_select = [doubles[i] for i in range(len(doubles)) if abs(grads[i]) > 1.0e-5]
# [[0, 1, 2, 3], [0, 1, 2, 9], [0, 1, 3, 8], [0, 1, 4, 5], [0, 1, 6, 7], [0, 1, 8, 9]]
print("Selected double excitation: ", doubles_select)

opt = qml.GradientDescentOptimizer(stepsize=0.5)

params_doubles = np.zeros(len(doubles_select), requires_grad=True)

for n in range(20):
    params_doubles = opt.step(cost_fn, params_doubles, excitations=doubles_select)

def circuit_2(params, wires, excitations, gates_select, params_select):
    qml.BasisState(hf_state, wires=wires)

    for i, gate in enumerate(gates_select):
        if len(gate) == 4:
            qml.DoubleExcitation(params_select[i], wires=gate)
        elif len(gate) == 2:
            qml.SingleExcitation(params_select[i], wires=gate)

    for i, gate in enumerate(excitations):
        if len(gate) == 4:
            qml.DoubleExcitation(params[i], wires=gate)
        elif len(gate) == 2:
            qml.SingleExcitation(params[i], wires=gate)

cost_fn = qml.ExpvalCost(circuit_2, H, dev, optimize=True)
circuit_gradient = qml.grad(cost_fn, argnum=0)
params = [0.0] * len(singles)

grads = circuit_gradient(
    params,
    excitations=singles,
    gates_select=doubles_select,
    params_select=params_doubles
)

for i in range(len(singles)):
    print(f"Excitation : {singles[i]}, Gradient: {grads[i]}")

singles_select = [singles[i] for i in range(len(singles)) if abs(grads[i]) > 1.0e-5]
# [[0, 2], [0, 8], [1, 3], [1, 9]]
print("Selected single excitation: ", singles_select)

cost_fn = qml.ExpvalCost(circuit_1, H, dev, optimize=True)

params = np.zeros(len(doubles_select + singles_select), requires_grad=True)

gates_select = doubles_select + singles_select

H_sparse = qml.utils.sparse_hamiltonian(H)
opt = qml.GradientDescentOptimizer(stepsize=0.5)

excitations = doubles_select + singles_select

params = np.zeros(len(excitations), requires_grad=True)

@qml.qnode(dev, diff_method="parameter-shift")
def circuit(params):
    qml.BasisState(hf_state, wires=range(qubits))

    for i, excitation in enumerate(excitations):
        if len(excitation) == 4:
            qml.DoubleExcitation(params[i], wires=excitation)
        elif len(excitation) == 2:
            qml.SingleExcitation(params[i], wires=excitation)

    return qml.expval(qml.SparseHamiltonian(H_sparse, wires=range(qubits)))


def cost(params):
    return circuit(params)


for n in range(100):
    t1 = time.time()
    params, energy = opt.step_and_cost(cost, params)
    t2 = time.time()
    print("n = {:},  E = {:.8f} H, t = {:.2f} s".format(n, energy, t2 - t1))

# n = 99,  E = -7.97240884 H, t = 0.14 s

print(f"Total number of gates: {len(singles_select)*3+len(doubles_select)*28}\n"
      f"Total number of two-qubit control gates: {len(singles_select)*3 + len(doubles_select)*14}")
# Total number of gates: 180
# Total number of two-qubit control gates: 96

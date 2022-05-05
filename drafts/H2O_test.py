import pennylane.numpy as np
from pennylane import qchem
import pennylane as qml
import time


#symbols, coordinates = qchem.read_structure("H2O.xyz")
symbols, coordinates =  ['H', 'O', 'H'], np.array([0.,0.,0.,1.63234543, 0.86417176, 0., 3.36087791, 0.,0.])*1.88973 # Angstrom to Bohr
print(symbols, coordinates)
# ['H', 'O', 'H']
# [0.         0.         0.         1.63234543 0.86417176 0. 3.36087791 0.         0.        ]
charge = 0
multiplicity = 1
basis_set = "sto-6g"
electrons = 10
orbitals = 7

import pyscf
from pyscf import gto

mol = pyscf.M(
    atom = 'H 0. 0. 0.; O 1.63234543 0.86417176 0; H 3.36087791 0. 0.  ',  # in Angstrom
    basis = basis_set,
    symmetry = True,
)
mol.unit = 'A'
#mol = gto.M(atom = 'H2O.xyz',basis = basis_set,symmetry = True)
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
"""
Results from code above:

converged SCF energy = -75.1645758157887
E(FCI) = -75.491788196432
converged SCF energy = -75.1645758157636  <S^2> = 4.0234216e-10  2S+1 = 1
E(UHF-FCI) = -75.491788196430
E(FCI) = -75.491788196432
"""

core, active = qchem.active_space(electrons, orbitals, active_electrons=4, active_orbitals=4)
print("List of core orbitals: {:}".format(core))
print("List of active orbitals: {:}".format(active))
print("Number of qubits: {:}".format(2 * len(active)))
H, qubits = qchem.molecular_hamiltonian(
    symbols,
    coordinates,
    charge=charge,
    mult=multiplicity,
    basis=basis_set,
    active_electrons=4,
    active_orbitals=4,
)

print("Number of qubits required to perform quantum simulations: {:}".format(qubits)) # 8
#print("Hamiltonian of the water molecule")
#print(H)
active_electrons = 4

singles, doubles = qchem.excitations(active_electrons, qubits)
print(f"Total number of excitations = {len(singles) + len(doubles)}")

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
# [[0, 1, 4, 5], [0, 1, 4, 7], [0, 1, 5, 6], [0, 1, 6, 7], [0, 2, 4, 6], [0, 3, 4, 5], [0, 3, 4, 7], [0, 3, 5, 6], [0, 3, 6, 7], [1, 2, 4, 5], [1, 2, 4, 7], [1, 2, 5, 6], [1, 2, 6, 7], [1, 3, 5, 7], [2, 3, 4, 5], [2, 3, 4, 7], [2, 3, 5, 6], [2, 3, 6, 7]]
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
# [[0, 4], [0, 6], [1, 5], [1, 7], [2, 4], [2, 6], [3, 5], [3, 7]]
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

print(f"Total number of gates: {len(singles_select)*3+len(doubles_select)*28}\n"
      f"Total number of two-qubit control gates: {len(singles_select)*3 + len(doubles_select)*14}")
# Total number of gates: 528
# Total number of two-qubit control gates: 276
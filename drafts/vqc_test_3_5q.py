# Modified from https://pennylane.ai/qml/demos/tutorial_vqls.html
# Pennylane
import pennylane as qml
from pennylane import numpy as np
# Plotting
import matplotlib.pyplot as plt
plt.style.use(['science','nature'])
import time

"""
A = zeta * I + J X_1 + J X_2 + gamma Z_2 Z_3  + eta Z_3 Z_4
"""

J = 0.1
zeta = 1
eta = 0.2
gamma = 0.1

n_layers =1
n_qubits = 5  # Number of system qubits.
n_shots = 10 ** 6  # Number of quantum measurements.
tot_qubits = n_qubits + 1  # Addition of an ancillary qubit.
ancilla_idx = n_qubits  # Index of the ancillary qubit (last position).
steps = 10 # Number of optimization steps
learning_rate = 2  # Learning rate
q_delta = 0.001  # Initial spread of random quantum weights
rng_seed =42 # Seed for random number generator

print("Number of Strongly Entangled Layers: ", n_layers)
# Coefficients of the linear combination A = c_0 A_0 + c_1 A_1 ...
c = np.array([zeta, J, J, gamma, eta])

def U_b():
    """Unitary matrix rotating the ground state to the problem vector |b> = U_b |0>."""
    for idx in range(n_qubits):
        qml.Hadamard(wires=idx)

def CA(idx):
    """Controlled versions of the unitary components A_l of the problem matrix A."""
    if idx == 0:
        # identity
        None

    elif idx == 1:
        # X_0
        qml.CNOT(wires=[ancilla_idx, 0])

    elif idx == 2:
        # X_1
        qml.CNOT(wires=[ancilla_idx, 1])

    elif idx == 3:
        # Z_1 Z_2
        qml.CZ(wires=[ancilla_idx, 1])
        qml.CZ(wires=[ancilla_idx, 2])

    elif idx == 4:
        # Z_2 Z_3
        qml.CZ(wires=[ancilla_idx, 2])
        qml.CZ(wires=[ancilla_idx, 3])


def variational_block(weights):
    """Variational circuit mapping the ground state |0> to the ansatz state |x>."""
    # We first prepare an equal superposition of all the states of the computational basis.
    for idx in range(n_qubits):
        qml.Hadamard(wires=idx)

    # A very minimal variational circuit.
    qml.StronglyEntanglingLayers(weights=weights, wires=range(n_qubits))

def vqls_circ_1():
    """
    for
    J = 0.1
    zeta = 1
    eta = 0.2
    :return:
    """
    for i in range(4):
        qml.Hadamard(wires=i)

    qml.Rot(0.5604857779729846,
                0.2975881694474745,
                -0.16130212489384718, wires=1)
    qml.Rot(-0.07462353342079096,
                0.33125436215663834,
                0.07889636851129728, wires=3)
    qml.Rot(0.05215167213498776,
                -1.4000847094560546e-07,
                -0.05215168141170644, wires=0)
    qml.CNOT(wires=[1,0])
    qml.Rot(0.039182651193984515,
                -0.29532920771860316,
                0.08795141616001576, wires=1)
    qml.CNOT(wires=[1,0])
    qml.Rot(0.0882697884073239,
                1.419131193023994e-07,
                -0.08826979432942605, wires=0)
    qml.Rot(-0.18223202170553732,
                -4.199805334992931e-09,
                0.1822320257843591, wires=2)
    qml.CNOT(wires=[2,3])
    qml.Rot(-0.6359327678530272,
                0.015263241646471063,
                0.10453369952015328, wires=1)
    #return qml.sample()

dev_mu = qml.device("default.qubit.autograd", wires=tot_qubits)

@qml.qnode(dev_mu, interface="autograd", diff_method="backprop")
def local_hadamard_test(weights, l=None, lp=None, j=None, part=None):

    # First Hadamard gate applied to the ancillary qubit.
    qml.Hadamard(wires=ancilla_idx)

    # For estimating the imaginary part of the coefficient "mu", we must add a "-i"
    # phase gate.
    if part == "Im" or part == "im":
        qml.PhaseShift(-np.pi / 2, wires=ancilla_idx)

    # Variational circuit generating a guess for the solution vector |x>
    variational_block(weights)

    # Controlled application of the unitary component A_l of the problem matrix A.
    CA(l)

    # Adjoint of the unitary U_b associated to the problem vector |b>.
    # In this specific example Adjoint(U_b) = U_b.
    U_b()

    # Controlled Z operator at position j. If j = -1, apply the identity.
    if j != -1:
        qml.CZ(wires=[ancilla_idx, j])

    # Unitary U_b associated to the problem vector |b>.
    U_b()

    # Controlled application of Adjoint(A_lp).
    # In this specific example Adjoint(A_lp) = A_lp.
    CA(lp)

    # Second Hadamard gate applied to the ancillary qubit.
    qml.Hadamard(wires=ancilla_idx)

    # Expectation value of Z for the ancillary qubit.
    return qml.expval(qml.PauliZ(wires=ancilla_idx))

def mu(weights, l=None, lp=None, j=None):
    """Generates the coefficients to compute the "local" cost function C_L."""

    mu_real = local_hadamard_test(weights, l=l, lp=lp, j=j, part="Re")
    mu_imag = local_hadamard_test(weights, l=l, lp=lp, j=j, part="Im")

    return mu_real + 1.0j * mu_imag

def psi_norm(weights):
    """Returns the normalization constant <psi|psi>, where |psi> = A |x>."""
    norm = 0.0

    for l in range(0, len(c)):
        for lp in range(0, len(c)):
            norm = norm + c[l] * np.conj(c[lp]) * mu(weights, l, lp, -1)

    return abs(norm)

def cost_loc(weights):
    """Local version of the cost function. Tends to zero when A|x> is proportional to |b>."""
    mu_sum = 0.0

    for l in range(0, len(c)):
        for lp in range(0, len(c)):
            for j in range(0, n_qubits):
                mu_sum = mu_sum + c[l] * np.conj(c[lp]) * mu(weights, l, lp, j)

    mu_sum = abs(mu_sum)

    # Cost function C_L
    return 0.5 - 0.5 * mu_sum / (n_qubits * psi_norm(weights))

np.random.seed(rng_seed)
w = q_delta * np.random.randn(n_qubits*n_layers*3, requires_grad=True).reshape((n_layers, n_qubits, 3))

opt = qml.GradientDescentOptimizer(learning_rate)

cost_history = []
#training_start = time.time()
for it in range(steps):
    epoch_start = time.time()
    w, cost = opt.step_and_cost(cost_loc, w)
    epoch_end = time.time()
    print("Step {:3d}       Cost_L = {:2.15f}    Time {:9.7f}".format(it, cost, epoch_end-epoch_start))
    cost_history.append(cost)

Id = np.identity(2)
Z = np.array([[1, 0], [0, -1]])
X = np.array([[0, 1], [1, 0]])
A_0 = np.identity(n_qubits**2)
A_1 = np.kron(X, np.kron(Id, np.kron(Id, Id)))
A_2 = np.kron(Id, np.kron(X, np.kron(Id, Id)))
A_3 = np.kron(Id, np.kron(Z, np.kron(Z, Id)))
A_4 = np.kron(Id, np.kron(Id, np.kron(Z, Z)))


A_num = c[0] * A_0 + c[1] * A_1 + c[2] * A_2 + c[3] * A_3 + c[4] * A_4
b = np.ones(2**n_qubits) / np.sqrt(2**n_qubits)

print("A = \n", A_num)
print("b = \n", b)

A_inv = np.linalg.inv(A_num)
x = np.dot(A_inv, b)

c_probs = (x / np.linalg.norm(x)) ** 2

dev_x = qml.device("default.qubit", wires=n_qubits, shots=n_shots)

@qml.qnode(dev_x)
def prepare_and_sample(weights):

    # Variational circuit generating a guess for the solution vector |x>
    variational_block(weights)

    # We assume that the system is measured in the computational basis.
    # then sampling the device will give us a value of 0 or 1 for each qubit (n_qubits)
    # this will be repeated for the total number of shots provided (n_shots)
    return qml.sample()


raw_samples = prepare_and_sample(w)

# convert the raw samples (bit strings) into integers and count them
samples = []
for sam in raw_samples:
    samples.append(int("".join(str(bs) for bs in sam), base=2))

q_probs = np.bincount(samples) / n_shots

print("x_n^2 =\n", c_probs)

print("|<x|n>|^2=\n", q_probs)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

ax1.bar(np.arange(0, 2 ** n_qubits), c_probs, color="lightgreen")
ax1.set_xlim(-0.5, 2 ** n_qubits - 0.5)
ax1.set_xlabel("Vector space basis")
ax1.set_title("Classical probabilities")

ax2.bar(np.arange(0, 2 ** n_qubits), q_probs, color="lightblue")
ax2.set_xlim(-0.5, 2 ** n_qubits - 0.5)
ax2.set_xlabel("Hilbert space basis")
ax2.set_title("Quantum probabilities")

plt.savefig('vqls_test_3_5q.png')
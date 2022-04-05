# Modified from https://pennylane.ai/qml/demos/tutorial_qaoa_maxcut.html
# Pennylane
import pennylane as qml
from pennylane import numpy as np
# Plotting
import matplotlib.pyplot as plt

import time

"""
The cost Hamiltonian:
C_alpha = 1/2 * (1 - Z_j Z_k)
(j, k) is an edge of the graph
Graph see https://pennylane.ai/qml/demos/tutorial_qaoa_maxcut.html
target solution is z = 1010
"""
n_wires = 4  # Number of system qubits.
graph = [(0,1), (0,3), (1,2), (2,3)]
n_shots = 100 # Number of quantum measurements.
steps = 50  # Number of optimization steps
learning_rate = 0.5  # Learning rate
q_delta = 0.001  # Initial spread of random quantum weights
rng_seed = 0  # Seed for random number generator
dev = qml.device("default.qubit", wires=n_wires, shots=1)
pauli_z = [[1, 0], [0, -1]]
pauli_z_2 = np.kron(pauli_z, pauli_z, requires_grad=False)

def bitstring_to_int(bit_string_sample):
    bit_string = "".join(str(bs) for bs in bit_string_sample)
    return int(bit_string, base=2)

def variational_block(weights):
    # A very minimal variational circuit.
    qml.StronglyEntanglingLayers(weights=weights, wires=range(n_wires))

@qml.qnode(dev)
def circuit(weights, edge=None):
    for wire in range(n_wires):
        qml.Hadamard(wires=wire)

    variational_block(weights)

    if edge is None:
        return qml.sample()

    return qml.expval(qml.Hermitian(pauli_z_2, wires=edge))

def qaoa_vqc(n_layers = 1):
    print("Number of Strongly Entangled Layers: ", n_layers)
    np.random.seed(rng_seed)
    init_params = q_delta * np.random.randn(n_wires*n_layers*3, requires_grad=True).reshape((n_layers, n_wires, 3))

    def objective(params):
        neg_obj = 0
        for edge in graph:
            neg_obj -= 0.5*(1-circuit(params, edge=edge))
        return neg_obj

    opt = qml.AdagradOptimizer(stepsize=learning_rate)

    params = init_params

    for i in range(steps):
        params = opt.step(objective, params)
        if (i + 1) % 5 == 0:
            print("Objective after step {:5d}: {: .7f}".format(i + 1, -objective(params)))

    bit_strings = []
    for i in range(0, n_shots):
        bit_strings.append(bitstring_to_int(circuit(params, edge=None)))

    counts = np.bincount(np.array(bit_strings))
    most_freq_bit_string = np.argmax(counts)
    print("Most frequently sampled bit string is: {:04b}".format(most_freq_bit_string))
    return -objective(params), bit_strings

bitstrings1 = qaoa_vqc(n_layers=1)[1]
bitstrings2 = qaoa_vqc(n_layers=2)[1]

xticks = range(0, 16)
xtick_labels = list(map(lambda x: format(x, "04b"), xticks))
bins = np.arange(0, 17) - 0.5

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title("n_layers=1")
plt.xlabel("bitstrings")
plt.ylabel("freq.")
plt.xticks(xticks, xtick_labels, rotation="vertical")
plt.hist(bitstrings1, bins=bins)
plt.subplot(1, 2, 2)
plt.title("n_layers=2")
plt.xlabel("bitstrings")
plt.ylabel("freq.")
plt.xticks(xticks, xtick_labels, rotation="vertical")
plt.hist(bitstrings2, bins=bins)
plt.tight_layout()
plt.savefig('qaoa_vqc_test.png')
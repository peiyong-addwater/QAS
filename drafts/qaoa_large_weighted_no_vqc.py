# Modified from https://pennylane.ai/qml/demos/tutorial_qaoa_maxcut.html
import pennylane as qml
from pennylane import numpy as np
import networkx as nx
from qiskit_optimization.applications import Maxcut, Tsp
import joblib
from collections import Counter, OrderedDict
np.random.seed(42)

n_wires = 7
graph =   [(0, 1, 0.1), (0, 2, 0.2),  (2, 3, 0.3), (1, 4, 0.4), (2, 4, 0.5), (0 ,5, 0.6),  (3, 6, 0.7), (1,6, 0.8)]
n_steps = 50
"""
The cost Hamiltonian:
C_alpha = 1/2 * (1 - Z_j Z_k)*w
(j, k) is an edge of the graph
optimal function value: 3.3
optimal value: [0. 1. 1. 1. 0. 1. 0.]
status: SUCCESS
energy: -1.5
max-cut objective: -3.3000000000000003
solution: [0 1 1 1 0 1 0]
solution objective: 3.3
"""
# unitary operator U_B with parameter beta
def U_B(beta):
    for wire in range(n_wires):
        qml.RX(2 * beta, wires=wire)


# unitary operator U_C with parameter gamma
def U_C(gamma):
    for edge in graph:
        wire1 = edge[0]
        wire2 = edge[1]
        qml.CNOT(wires=[wire1, wire2])
        qml.RZ(gamma, wires=wire2)
        qml.CNOT(wires=[wire1, wire2])

def bitstring_to_int(bit_string_sample):
    bit_string = "".join(str(bs) for bs in bit_string_sample)
    return int(bit_string, base=2)
def sample_result_to_str(bit_string_sample):
    return "".join(str(bs) for bs in bit_string_sample)
dev = qml.device("default.qubit", wires=n_wires, shots=1)
pauli_z = [[1, 0], [0, -1]]
pauli_z_2 = np.kron(pauli_z, pauli_z, requires_grad=False)


@qml.qnode(dev)
def circuit(gammas, betas, edge=None, n_layers=1):
    # apply Hadamards to get the n qubit |+> state
    for wire in range(n_wires):
        qml.Hadamard(wires=wire)
    # p instances of unitary operators
    for i in range(n_layers):
        U_C(gammas[i])
        U_B(betas[i])
    if edge is None:
        # measurement phase
        return qml.sample()
    # during the optimization phase we are evaluating a term
    # in the objective using expval
    wires = [edge[0], edge[1]]
    return qml.expval(qml.Hermitian(pauli_z_2, wires=wires))

def qaoa_maxcut(n_layers=1):
    print("\np={:d}".format(n_layers))

    # initialize the parameters near zero
    init_params = 0.01 * np.random.rand(2, n_layers, requires_grad=True)

    # minimize the negative of the objective function
    def objective(params):
        gammas = params[0]
        betas = params[1]
        neg_obj = 0
        for edge in graph:
            weight = edge[2]
            # objective for the MaxCut problem
            neg_obj -= 0.5 * (1 - circuit(gammas, betas, edge=edge, n_layers=n_layers))*weight
        return neg_obj

    # initialize optimizer: Adagrad works well empirically
    opt = qml.AdagradOptimizer(stepsize=0.1)

    # optimize parameters in objective
    params = init_params
    for i in range(n_steps):
        params = opt.step(objective, params)
        if (i + 1) % 5 == 0:
            print("Objective after step {:5d}: {: .7f}".format(i + 1, -objective(params)))

    # sample measured bitstrings 100 times
    bit_strings = []
    original_samples = []
    n_samples = 10000
    for i in range(0, n_samples):
        bits = circuit(params[0], params[1], edge=None, n_layers=n_layers)
        bit_strings.append(bitstring_to_int(bits))
        original_samples.append(sample_result_to_str(bits))

    # print optimal parameters and most frequently sampled bitstring
    counts = np.bincount(np.array(bit_strings))
    most_freq_bit_string = np.argmax(counts)
    original_samples = Counter(original_samples)
    original_samples = dict(OrderedDict(original_samples.most_common()))
    print(original_samples)
    print("Optimized (gamma, beta) vectors:\n{}".format(params[:, :n_layers]))
    print("Most frequently sampled bit string is: {:07b}".format(most_freq_bit_string))

    return -objective(params), bit_strings, original_samples


# perform qaoa on our graph with p=1,2 and
# keep the bitstring sample lists
p1 = 1
p2 = 2
bitstrings1, os1 = qaoa_maxcut(n_layers=p1)[1:]
bitstrings2, os2 = qaoa_maxcut(n_layers=p2)[1:]
print("{} layer(s): \n".format(p1), os1)
print("{} layer(s): \n".format(p2), os2)
"""
import matplotlib.pyplot as plt

xticks = range(0, 2**7)
xtick_labels = list(map(lambda x: format(x, "07b"), xticks))
bins = np.arange(0, 2**7+1) - 0.5

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title("n_layers={}".format(p1))
plt.xlabel("bitstrings")
plt.ylabel("freq.")
plt.xticks(xticks, xtick_labels, rotation="vertical")
plt.hist(bitstrings1, bins=bins)
plt.subplot(1, 2, 2)
plt.title("n_layers={}".format(p2))
plt.xlabel("bitstrings")
plt.ylabel("freq.")
plt.xticks(xticks, xtick_labels, rotation="vertical")
plt.hist(bitstrings2, bins=bins)
plt.tight_layout()
plt.savefig('qaoa_large_test_no_vqc.png')
"""
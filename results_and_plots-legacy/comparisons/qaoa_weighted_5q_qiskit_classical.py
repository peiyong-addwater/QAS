# useful additional packages
import matplotlib.pyplot as plt
import matplotlib.axes as axes
import numpy as np
import networkx as nx
import operator
from qiskit import Aer
from qiskit.tools.visualization import plot_histogram
from qiskit.circuit.library import TwoLocal
from qiskit_optimization.applications import Maxcut, Tsp
from qiskit.algorithms import VQE, NumPyMinimumEigensolver
from qiskit.algorithms.optimizers import SPSA
from qiskit.utils import algorithm_globals, QuantumInstance
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.problems import QuadraticProgram
from collections import OrderedDict

"""
Offset: -10.5
Ising Hamiltonian:
0.5 * IIIZZ
+ 1.0 * IIZIZ
+ 1.5 * IZZII
+ 3.0 * ZIIIZ
+ 2.0 * ZIIZI
+ 2.5 * ZIZII
optimal function value: 18.0
optimal value: [0. 0. 0. 1. 1.]
status: SUCCESS
energy: -7.5
max-cut objective: -18.0
solution: [0 0 0 1 1]
solution objective: 18.0
"""

n = 5  # Number of nodes in graph
G = nx.Graph()
G.add_nodes_from(np.arange(0, n, 1))
elist = [(0, 1, 1), (0, 2, 2),  (2, 3, 3), (1, 4, 4), (2, 4, 5), (0, 4, 6)]
G.add_weighted_edges_from(elist)

colors = ["#00b4d9" for node in G.nodes()]
pos = nx.spring_layout(G,seed=42)


def draw_graph(G, colors, pos, save_name = 'qaoa_5q_weighted_test.png'):
    default_axes = plt.axes(frameon=False)
    nx.draw_networkx(G, node_color=colors, node_size=300, alpha=0.8, pos=pos, ax=default_axes)
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)
    plt.savefig(save_name)


draw_graph(G, colors, pos, 'fig_max_cut_weighted_5q.pdf')

sol_1 = list("11100")
sol_2 = list("00011")
colors_1 = ["r" if sol_1[i] == "0" else "c" for i in range(n)]
draw_graph(G, colors_1, pos, 'fig_maxcut_5q_res_11100.pdf')

colors_2 = ["r" if sol_2[i] == "0" else "c" for i in range(n)]
draw_graph(G, colors_2, pos, 'fig_maxcut_5q_res_00011.pdf')

# Computing the weight matrix from the random graph
w = np.zeros([n, n])
for i in range(n):
    for j in range(n):
        temp = G.get_edge_data(i, j, default=0)
        if temp != 0:
            w[i, j] = temp["weight"]
print(w)

best_cost_brute = 0
cases = {}
for b in range(2**n):
    x = [int(t) for t in reversed(list(bin(b)[2:].zfill(n)))]
    cost = 0
    for i in range(n):
        for j in range(n):
            cost = cost + w[i, j] * x[i] * (1 - x[j])
    if best_cost_brute < cost:
        best_cost_brute = cost
        xbest_brute = x
    print("case = " + str(x) + " cost = " + str(cost))
    cases[str(x)] = cost
cases = dict(sorted(cases.items(), key=operator.itemgetter(1),reverse=True))
print("All Solutions:\n", cases)
colors = ["r" if xbest_brute[i] == 0 else "c" for i in range(n)]
draw_graph(G, colors, pos, 'max_cut_5q_weighted_solution.png')
print("\nBest solution = " + str(xbest_brute) + " cost = " + str(best_cost_brute))

max_cut = Maxcut(w)
qp = max_cut.to_quadratic_program()
print(qp.export_as_lp_string())

qubitOp, offset = qp.to_ising()
print("Offset:", offset)
print("Ising Hamiltonian:")
print(str(qubitOp))

# solving Quadratic Program using exact classical eigensolver
exact = MinimumEigenOptimizer(NumPyMinimumEigensolver())
result = exact.solve(qp)
print(result)

# Making the Hamiltonian in its full form and getting the lowest eigenvalue and eigenvector
ee = NumPyMinimumEigensolver()
result = ee.compute_minimum_eigenvalue(qubitOp)

x = max_cut.sample_most_likely(result.eigenstate)
print("energy:", result.eigenvalue.real)
print("max-cut objective:", result.eigenvalue.real + offset)
print("solution:", x)
print("solution objective:", qp.objective.evaluate(x))

colors = ["r" if x[i] == 0 else "c" for i in range(n)]
draw_graph(G, colors, pos, 'qaoa_5q_weighted_test.png')


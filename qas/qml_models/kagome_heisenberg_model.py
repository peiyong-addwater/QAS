# Model parameters from https://github.com/qiskit-community/open-science-prize-2022/blob/main/kagome-vqe.ipynb
# A 12-site kagome lattice on a 16-qubit quantum processor

import shutup
shutup.please()

t = 1.0
kagome_edge_list = [
    (1, 2, t),
    (2, 3, t),
    (3, 5, t),
    (5, 8, t),
    (8, 11, t),
    (11, 14, t),
    (14, 13, t),
    (13, 12, t),
    (12, 10, t),
    (10, 7, t),
    (7, 4, t),
    (4, 1, t),
    (4, 2, t),
    (2, 5, t),
    (5, 11, t),
    (11, 13, t),
    (13, 10, t),
    (10, 4, t),
]


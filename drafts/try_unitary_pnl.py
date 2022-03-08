import pennylane as qml
from pennylane import numpy as np
def circuit(theta):
    qml.RX(theta, wires=1)
    qml.PauliZ(wires=0)

get_matrix = qml.transforms.get_unitary_matrix(circuit)
print(get_matrix(0.5))
real_part = np.real(get_matrix(0.5))
print(real_part)
img_part = np.imag(get_matrix(0.5))
print(img_part)

def cost_func(x):
    mat = get_matrix(x)
    r = np.real(mat)
    im = np.imag(mat)
    target = np.eye(4)
    r_diff = np.linalg.norm(target-r)
    im_diff = np.linalg.norm(im)
    return r_diff+im_diff

print(cost_func(0.5))

cost_grad = qml.grad(cost_func)
print(cost_grad(0.5))

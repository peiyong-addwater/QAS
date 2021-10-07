import pennylane as qml 
import pennylane.numpy as pnp 
import numpy as np 
from qas import qml_gate_ops
from qas import qml_models 
from qas import mcts
import os 
import json 
from pprint import pprint

dev = qml.device('default.qubit', wires=3)


cwd = os.getcwd() 
print(cwd)

res_file = "20211005-114848.json"
with open(os.path.join(cwd, res_file)) as f:
    res_dict = json.load(f)
print(res_dict.keys())
print(res_dict['pool'])
pool = {}
for key, gate_dict in res_dict['pool'].items():
    pool[int(key)] = gate_dict

print(pool)
full_params = np.array(res_dict['params'])
p,c,l = full_params.shape[0], full_params.shape[1], full_params.shape[2]
k = res_dict['k']
toffoli_model = qml_models.ToffoliQMLSwapTestNoiseless(p,c,l, k, pool)
param_indices = toffoli_model.param_indices
extracted_params = [full_params[index] for index in param_indices]
example_circ = toffoli_model.constructFullCirc()
drawer = qml.draw(example_circ)
print(drawer(extracted_params))
print(example_circ(extracted_params))

print(toffoli_model.getLoss(full_params))
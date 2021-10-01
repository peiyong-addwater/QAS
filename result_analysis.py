import pennylane as qml 
import pennylane.numpy as pnp 
import numpy as np 
from qas import qml_ops 
from qas import qml_models 
from qas import mcts
import os 
import json 
from pprint import pprint

dev = qml.device('default.qubit', wires=3)


cwd = os.getcwd() 
print(cwd)

res_file = "20211001-025711.json"
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
toffoli_model = qml_models.PhaseFlipQMLNoiseless(p,c,l, k, pool)
toffoli_model2 = qml_models.ToffoliQMLNoiseless(p,c,l, k, pool)
param_indices = toffoli_model.param_indices
extracted_params = [full_params[index] for index in param_indices]
example_circ = toffoli_model.constructFullCirc()
drawer = qml.draw(example_circ)
print(drawer(extracted_params, x=toffoli_model.x_list[0], y = toffoli_model.y_list[0]))
print(example_circ(extracted_params, x=toffoli_model.x_list[0], y = toffoli_model.y_list[0]))



print()
print(toffoli_model.toList(full_params))
print()
print(res_dict['op_list'])
print()
print(toffoli_model.getLoss(full_params))
print(toffoli_model.getReward(full_params))
# print(toffoli_model.getLoss(full_params)+(toffoli_model.getReward(full_params)))
"""
print(toffoli_model2.getLoss(full_params))
print(toffoli_model2.getReward(full_params))
print(toffoli_model2.getLoss(full_params)+(toffoli_model.getReward(full_params)))
"""



final_params, loss_list = mcts.circuitModelTuning(
        initial_params=full_params,
        model=qml_models.ToffoliQMLNoiseless,
        num_epochs=100,
        k=res_dict['k'],
        op_pool=pool,
        opt_callable=qml.AdamOptimizer,
        lr=0.1,
        grad_noise_factor=0,
        verbose=1
    )

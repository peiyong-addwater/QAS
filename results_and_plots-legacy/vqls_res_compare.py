import numpy as np
import os
import json

vqls_4q_filename = "20220408-040418_VQLSDemo_4Q_QMLStateBasicGates.json"
cwd = os.getcwd()
with open(os.path.join(cwd, vqls_4q_filename)) as f:
    vqls_4q_res_dict = json.load(f)

quantum_res = np.array(vqls_4q_res_dict['quantum_result'])
classical_res = np.array(vqls_4q_res_dict['classical_result'])

error = np.mean(np.abs(quantum_res-classical_res)**2)
print(error) # 9.852235208181145e-06
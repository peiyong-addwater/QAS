import numpy as np
import os
import json
import matplotlib.pyplot as plt
plt.style.use(['science','nature'])
cwd = os.getcwd()
print(cwd)

"""
422 results
"""
res_1_file_422 = 'res-20211012-020301.json'
res_2_file_422 = 'res-20211012-131536.json'
with open(os.path.join(cwd, res_1_file_422)) as f:
    res_422_dict_1 = json.load(f)

with open(os.path.join(cwd, res_2_file_422)) as f:
    res_422_dict_2 = json.load(f)

print(res_422_dict_1.keys())
print(res_422_dict_2.keys())
reward_list_1 = [c[2] for c in res_422_dict_1['search_reward_list']]
reward_list_2 = [c[2] for c in res_422_dict_2['search_reward_list']]
fig = plt.figure()
plt.plot(list(range(len(reward_list_1))), reward_list_1,linestyle = '-',marker = 'x', label = 'Circuit a')
plt.plot(list(range(len(reward_list_2))), reward_list_2,linestyle = '-',marker = '.', label = 'Circuit b')
plt.xticks(list(range(len(reward_list_2))))
#plt.title("Search Reward for [[4,2,2]] Code Encoding Circuit (1)")
plt.xlabel('Epoch')
plt.ylabel('Reward')
plt.legend()
plt.savefig('fig_422_rewards_1_2.pdf')

"""
H2 results
"""
h2_with_only_neighbouring_cnots = 'res-20211123-113657.json'
h2_with_all_cnots = 'res-20211123-025029.json'
with open(os.path.join(cwd, h2_with_only_neighbouring_cnots)) as f:
    neighbouring_cnots = json.load(f)

with open(os.path.join(cwd, h2_with_all_cnots)) as f:
    all_cnots = json.load(f)

E_fci = -1.136189454088
Min_Energy = E_fci

nei_cnots_search_rewards = [s[2] for s in neighbouring_cnots['search_reward_list']]
neigh_cnots_fine_tune_loss = neighbouring_cnots['fine_tune_loss']
all_cnots_search_rewards = [s[2] for s in all_cnots['search_reward_list']]
all_cnots_fine_tune_loss = all_cnots['fine_tune_loss']

fig = plt.figure()
plt.plot(list(range(len(nei_cnots_search_rewards))), nei_cnots_search_rewards,marker = 'x')
plt.xlabel('Epoch')
plt.ylabel('Reward(-Energy, Ha)')
#plt.title("Search Reward with Only Neighbouring CNOTs")
plt.legend()
plt.savefig('fig_nei_cnots_search_rewards.pdf')

fig = plt.figure()
plt.plot(list(range(len(neigh_cnots_fine_tune_loss))), neigh_cnots_fine_tune_loss,label = r"$E_\mathrm{Search}$",linestyle = '-',marker = 'x')
plt.axhline(y = Min_Energy, color = 'r', linestyle = '--',label = r"$E_\mathrm{FCI}=-1.136189454088 Ha$")
#plt.title("Fine-tune Loss after Searching with Only Neighbouring CNOTs")
plt.xlabel('Epoch')
plt.ylabel('Loss (Energy, Ha)')
plt.legend()
plt.savefig('fig_neigh_cnots_fine_tune_loss.pdf')

fig = plt.figure()
plt.plot(list(range(len(all_cnots_search_rewards))), all_cnots_search_rewards,marker = 'x')
#plt.title("Search Reward without Restrictions on CNOT Locations")
plt.xlabel('Epoch')
plt.ylabel('Reward(-Energy)')
plt.savefig('fig_all_cnots_search_rewards.pdf')

fig = plt.figure()
plt.plot(list(range(len(all_cnots_fine_tune_loss))), all_cnots_fine_tune_loss,label = r"$E_\mathrm{Search}$",linestyle = '-',marker = 'x')
plt.axhline(y = Min_Energy, color = 'r', linestyle = '--',label = r"$E_\mathrm{FCI}=-1.136189454088Ha$")
#plt.title("Fine-tune Loss after Search, No Restrictions on CNOT Locations")
plt.xlabel('Epoch')
plt.ylabel('Loss (Energy, Ha)')
plt.savefig('fig_all_cnots_fine_tune_loss.pdf')

"""
QAOA results
"""
qaoa_res_file_1 = '20220406-103640_QAOAVQCDemo_4Q_QMLStateBasicGates.json'
qaoa_res_file_2 = '20220406-155957_QAOAVQCDemo_4Q_QMLStateBasicGates.json'
with open(os.path.join(cwd, qaoa_res_file_1)) as f:
    qaoa_dict_1 = json.load(f)
with open(os.path.join(cwd, qaoa_res_file_2)) as f:
    qaoa_dict_2 = json.load(f)

qaoa_early_stopping = 3.9
qaoa_1_search_reward_list = [s[2] for s in qaoa_dict_1['search_reward_list']]
qaoa_1_fine_tune_loss = qaoa_dict_1['fine_tune_loss']
qaoa_1_measurement_result = qaoa_dict_1['quantum_result'][1]

qaoa_2_search_reward_list = [s[2] for s in qaoa_dict_2['search_reward_list']]
qaoa_2_fine_tune_loss = qaoa_dict_2['fine_tune_loss']
qaoa_2_measurement_result = qaoa_dict_2['quantum_result'][1]

fig = plt.figure()
plt.plot(list(range(len(qaoa_1_search_reward_list))), qaoa_1_search_reward_list,marker = 'x')
plt.xlabel('Epoch')
plt.ylabel('Reward(-Objective)')
plt.axhline(y = qaoa_early_stopping, color = 'r', linestyle = '--',label = r"Early Stopping at 3.9")
plt.legend()
plt.savefig('fig_qaoa_1_search_rewards.pdf')

fig = plt.figure()
plt.plot(list(range(len(qaoa_2_search_reward_list))), qaoa_2_search_reward_list,marker = 'x')
plt.xlabel('Epoch')
plt.ylabel('Reward(-Objective)')
plt.axhline(y = qaoa_early_stopping, color = 'r', linestyle = '--',label = r"Early Stopping at 3.9")
plt.legend()
plt.savefig('fig_qaoa_2_search_rewards.pdf')

fig = plt.figure()
plt.plot(list(range(len(qaoa_1_fine_tune_loss))), qaoa_1_fine_tune_loss,linestyle = '-',marker = 'x')
plt.axhline(y = -4, color = 'r', linestyle = '--',label = r"Objective at optimal solution")
#plt.title("Fine-tune Loss after Searching with Only Neighbouring CNOTs")
plt.xlabel('Epoch')
plt.ylabel('Loss (Energy, Ha)')
plt.legend()
plt.savefig('fig_qaoa_1_fine_tune_loss.pdf')

fig = plt.figure()
plt.plot(list(range(len(qaoa_2_fine_tune_loss))), qaoa_2_fine_tune_loss,linestyle = '-',marker = 'x')
plt.axhline(y = -4, color = 'r', linestyle = '--',label = r"Objective at optimal solution")
#plt.title("Fine-tune Loss after Searching with Only Neighbouring CNOTs")
plt.xlabel('Epoch')
plt.ylabel('Loss (Energy, Ha)')
plt.legend()
plt.savefig('fig_qaoa_2_fine_tune_loss.pdf')

xticks = range(0, 16)
xtick_labels = list(map(lambda x: format(x, "04b"), xticks))
bins = np.arange(0, 17) - 0.5
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title("Quantum Result 0101")
plt.xlabel("Bitstrings")
plt.ylabel("Frequency")
plt.xticks(xticks, xtick_labels, rotation="vertical")
plt.hist(qaoa_1_measurement_result, bins=bins)
plt.subplot(1, 2, 2)
plt.title("Quantum Result 1010")
plt.xlabel("Bitstrings")
plt.ylabel("Frequency")
plt.xticks(xticks, xtick_labels, rotation="vertical")
plt.hist(qaoa_2_measurement_result, bins=bins)
plt.tight_layout()
plt.savefig('fig_qaoa_search_measurements.pdf')



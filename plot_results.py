import numpy as np
import os
import json
import matplotlib.pyplot as plt
import pennylane as qml
import networkx as nx
plt.style.use(['science','nature'])
cwd = os.getcwd()
#print(cwd)

"""
422 results
"""
res_1_file_422 = 'res-20211012-020301.json'
res_2_file_422 = 'res-20211012-131536.json'
with open(os.path.join(cwd, res_1_file_422)) as f:
    res_422_dict_1 = json.load(f)

with open(os.path.join(cwd, res_2_file_422)) as f:
    res_422_dict_2 = json.load(f)

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
plt.close()

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
plt.close()

fig = plt.figure()
plt.plot(list(range(len(neigh_cnots_fine_tune_loss))), neigh_cnots_fine_tune_loss,label = r"$E_\mathrm{Search}$",linestyle = '-',marker = 'x')
plt.axhline(y = Min_Energy, color = 'r', linestyle = '--',label = r"$E_\mathrm{FCI}=-1.136189454088 Ha$")
#plt.title("Fine-tune Loss after Searching with Only Neighbouring CNOTs")
plt.xlabel('Epoch')
plt.ylabel('Loss (Energy, Ha)')
plt.legend()
plt.savefig('fig_neigh_cnots_fine_tune_loss.pdf')
plt.close()

fig = plt.figure()
plt.plot(list(range(len(all_cnots_search_rewards))), all_cnots_search_rewards,marker = 'x')
#plt.title("Search Reward without Restrictions on CNOT Locations")
plt.xlabel('Epoch')
plt.ylabel('Reward(-Energy)')
plt.savefig('fig_all_cnots_search_rewards.pdf')
plt.close()

fig = plt.figure()
plt.plot(list(range(len(all_cnots_fine_tune_loss))), all_cnots_fine_tune_loss,label = r"$E_\mathrm{Search}$",linestyle = '-',marker = 'x')
plt.axhline(y = Min_Energy, color = 'r', linestyle = '--',label = r"$E_\mathrm{FCI}=-1.136189454088Ha$")
#plt.title("Fine-tune Loss after Search, No Restrictions on CNOT Locations")
plt.xlabel('Epoch')
plt.ylabel('Loss (Energy, Ha)')
plt.savefig('fig_all_cnots_fine_tune_loss.pdf')
plt.close()

"""
LiH Results
"""
Min_energy_LiH = -7.8825378193
lih_early_stopping = 7.6
lih_results_file = '20220430-043910_LiH_QMLStateBasicGates.json'
with open(os.path.join(cwd, lih_results_file)) as f:
    lih_results = json.load(f)

lih_search_rewards = [s[2] for s in lih_results['search_reward_list']]
lih_finetuen_rewards = lih_results['fine_tune_loss']
fig = plt.figure()
plt.plot(list(range(len(lih_search_rewards))), lih_search_rewards, marker = 'x')
plt.axhline(y = lih_early_stopping, color = 'r', linestyle = '--',label = r"Early Stopping at {}".format(lih_early_stopping))
plt.xlabel('Epoch')
plt.ylabel('Reward(-Energy, Ha)')
plt.legend()
plt.savefig('fig_LiH_search_rewards.pdf')

fig = plt.figure()
plt.plot(list(range(len(lih_finetuen_rewards))), lih_finetuen_rewards,label = r"$E_\mathrm{Search}$",linestyle = '-',marker = 'x')
plt.axhline(y = Min_energy_LiH, color = 'r', linestyle = '--',label = r"$E_\mathrm{FCI}=-7.8825 Ha$")
#plt.title("Fine-tune Loss after Searching with Only Neighbouring CNOTs")
plt.xlabel('Epoch')
plt.ylabel('Loss (Energy, Ha)')
plt.legend()
plt.savefig('fig_LiH_fine_tune_loss.pdf')
plt.close()


"""
QAOA results
"""
n_new_samples = 1000

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
plt.close()

fig = plt.figure()
plt.plot(list(range(len(qaoa_2_search_reward_list))), qaoa_2_search_reward_list,marker = 'x')
plt.xlabel('Epoch')
plt.ylabel('Reward(-Objective)')
plt.axhline(y = qaoa_early_stopping, color = 'r', linestyle = '--',label = r"Early Stopping at 3.9")
plt.legend()
plt.savefig('fig_qaoa_2_search_rewards.pdf')
plt.close()

fig = plt.figure()
plt.plot(list(range(len(qaoa_1_fine_tune_loss))), qaoa_1_fine_tune_loss,linestyle = '-',marker = 'x')
plt.axhline(y = -4, color = 'r', linestyle = '--',label = r"Objective at optimal solution")
#plt.title("Fine-tune Loss after Searching with Only Neighbouring CNOTs")
plt.xlabel('Epoch')
plt.ylabel('Loss (Energy, Ha)')
plt.legend()
plt.savefig('fig_qaoa_1_fine_tune_loss.pdf')
plt.close()

fig = plt.figure()
plt.plot(list(range(len(qaoa_2_fine_tune_loss))), qaoa_2_fine_tune_loss,linestyle = '-',marker = 'x')
plt.axhline(y = -4, color = 'r', linestyle = '--',label = r"Objective at optimal solution")
#plt.title("Fine-tune Loss after Searching with Only Neighbouring CNOTs")
plt.xlabel('Epoch')
plt.ylabel('Loss (Energy, Ha)')
plt.legend()
plt.savefig('fig_qaoa_2_fine_tune_loss.pdf')
plt.close()

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
plt.savefig('fig_qaoa_search_measurements_after_search.pdf')
plt.close()


dev_qaoa_4_qubit = qml.device('lightning.qubit', wires=(0, 1, 2, 3), shots=1)

def bitstring_to_int(bit_string_sample):
    bit_string = "".join(str(bs) for bs in bit_string_sample)
    return int(bit_string, base=2)

@qml.qnode(dev_qaoa_4_qubit)
def qaoa_circuit_1():
    for wire in range(4):
        qml.Hadamard(wires=wire)
    qml.U3(1.5707963267948533, -0.3618239485577399, -2.480977337366047e-06, wires=0)
    qml.U3(1.5707963267955087, -0.5492239787830435, 5.473638905667877e-13, wires=3)
    qml.CNOT(wires=[3,2])
    qml.CNOT(wires=[2,3])
    qml.CNOT(wires=[3,0])
    qml.CNOT(wires=[0,2])
    qml.U3(1.570796326794654, 0.5482710337796082, 1.372802127183914e-13, wires=1)
    qml.CNOT(wires=[0,3])
    qml.U3(-1.5707963267947023, 0.21428943152379354, 1.5149065173482657e-13, wires=0)
    return qml.sample()

@qml.qnode(dev_qaoa_4_qubit)
def qaoa_circuit_2():
    for wire in range(4):
        qml.Hadamard(wires=wire)
    qml.U3(1.570796326794232, 0.9690786345916832, 3.314772140674725e-13, wires=3)
    qml.U3(1.570796326793921, -0.04717642396316069, -2.046468624499896e-13, wires=0)
    qml.CNOT(wires=[0, 3])
    qml.U3(1.5707963267955454, -0.7192439786470295, -1.2206061490770083e-13, wires=2)
    qml.U3(-1.0352228967671575, -0.22822101133807382, -0.41436697575645365, wires=1)
    qml.CNOT(wires=[1, 0])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[3, 2])
    qml.CNOT(wires=[1, 3])
    return qml.sample()


bitstrings1, bitstrings2 = [], []
for i in range(n_new_samples):
    bitstrings1.append(bitstring_to_int(qaoa_circuit_1()))
    bitstrings2.append(bitstring_to_int(qaoa_circuit_2()))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))
plt.subplot(1, 2, 1)
plt.title("Quantum Result 0101")
plt.xlabel("Bitstrings")
plt.ylabel("Frequency")
plt.xticks(xticks, xtick_labels, rotation="vertical")
plt.hist(bitstrings1, bins=bins)
plt.subplot(1, 2, 2)
plt.title("Quantum Result 1010")
plt.xlabel("Bitstrings")
plt.ylabel("Frequency")
plt.xticks(xticks, xtick_labels, rotation="vertical")
plt.hist(bitstrings2, bins=bins)
plt.tight_layout()
plt.savefig('fig_qaoa_search_measurements_more_samples.pdf')
plt.close()

"""
QAOA 5 Qubit
"""
qaoa_5q_early_stopping = 16.5

qaoa_5q_1_filename = "20220429-044101_QAOAWeightedVQCDemo_5Q_QMLStateBasicGates.json"
with open(os.path.join(cwd, qaoa_5q_1_filename)) as f:
    qaoa_dict_5q_1 = json.load(f)

qaoa_5q_search_reward_list_1 = [s[2] for s in qaoa_dict_5q_1['search_reward_list']]
qaoa_5q_finetune_loss_1 = qaoa_dict_5q_1['fine_tune_loss']

fig = plt.figure()
plt.plot(list(range(len(qaoa_5q_search_reward_list_1))), qaoa_5q_search_reward_list_1, marker = 'x')
plt.xlabel('Epoch')
plt.ylabel('Reward(-Objective)')
plt.axhline(y = qaoa_5q_early_stopping, color = 'r', linestyle = '--',label = r"Early Stopping at {}".format(qaoa_5q_early_stopping))
plt.legend()
plt.savefig('fig_qaoa_5q_1_search_rewards.pdf')
plt.close()

fig = plt.figure()
plt.plot(list(range(len(qaoa_5q_finetune_loss_1))), qaoa_5q_finetune_loss_1,linestyle = '-',marker = 'x')
plt.axhline(y = -18, color = 'r', linestyle = '--',label = r"Objective at optimal solution")
#plt.title("Fine-tune Loss after Searching with Only Neighbouring CNOTs")
plt.xlabel('Epoch')
plt.ylabel('Loss (Energy)')
plt.legend()
plt.savefig('fig_qaoa_5q_1_fine_tune_loss.pdf')
plt.close()

dev_qaoa_5_qubit = qml.device('lightning.qubit', wires=(0, 1, 2, 3, 4), shots=1)
@qml.qnode(dev_qaoa_5_qubit)
def qaoa_circuit_5q_1():
    for wire in range(5):
        qml.Hadamard(wires=wire)

    qml.Rot(-0.38173845607538814,
                0.9067659463333607,
                0.3814142580159696, wires=4)
    qml.CNOT(wires=[4,0])
    qml.Rot(0.19591582037207156,
                0.7512618503754793,
                0.20408548408219923, wires=4)
    qml.Rot(0.00010965282979182945,
                -1.5707963268102163,
                -0.9936554657183598, wires=1)
    qml.Rot(1.7677290917417184e-13,
                1.5707963267948857,
                -0.05531231276375175, wires=0)
    qml.CNOT(wires=[4,0])
    qml.CNOT(wires=[1,2])
    qml.Rot(1.447953629484945e-13,
                -1.5707963267907148,
                -0.16760111836032035, wires=2)
    qml.Rot(-8.642119045772874e-13,
                1.570796326796216,
                -0.10867496585753135, wires=3)
    qml.CNOT(wires=[2,3])
    return qml.sample()



"""
QAOA 7 Qubit
"""
qaoa_7q_1_filename = "20220409-091016_QAOAVQCDemo_7Q_QMLStateBasicGates.json"
qaoa_7q_2_filename = "20220409-091019_QAOAVQCDemo_7Q_QMLStateBasicGates.json"
with open(os.path.join(cwd, qaoa_7q_1_filename)) as f:
    qaoa_dict_7q_1 = json.load(f)
with open(os.path.join(cwd, qaoa_7q_2_filename)) as f:
    qaoa_dict_7q_2 = json.load(f)

qaoa_7q_early_stopping = 6.5

qaoa_7q_1_search_reward_list = [s[2] for s in qaoa_dict_7q_1['search_reward_list']]
qaoa_7q_1_fine_tune_loss = qaoa_dict_7q_1['fine_tune_loss']

qaoa_7q_2_search_reward_list = [s[2] for s in qaoa_dict_7q_2['search_reward_list']]
qaoa_7q_2_fine_tune_loss = qaoa_dict_7q_2['fine_tune_loss']

fig = plt.figure()
plt.plot(list(range(len(qaoa_7q_1_search_reward_list))), qaoa_7q_1_search_reward_list,marker = 'x')
plt.xlabel('Epoch')
plt.ylabel('Reward(-Objective)')
plt.axhline(y = qaoa_7q_early_stopping, color = 'r', linestyle = '--',label = r"Early Stopping at {}".format(qaoa_7q_early_stopping))
plt.legend()
plt.savefig('fig_qaoa_7q_1_search_rewards.pdf')
plt.close()

fig = plt.figure()
plt.plot(list(range(len(qaoa_7q_2_search_reward_list))), qaoa_7q_2_search_reward_list,marker = 'x')
plt.xlabel('Epoch')
plt.ylabel('Reward(-Objective)')
plt.axhline(y = qaoa_7q_early_stopping, color = 'r', linestyle = '--',label = r"Early Stopping at {}".format(qaoa_7q_early_stopping))
plt.legend()
plt.savefig('fig_qaoa_7q_2_search_rewards.pdf')
plt.close()

fig = plt.figure()
plt.plot(list(range(len(qaoa_7q_1_fine_tune_loss))), qaoa_7q_1_fine_tune_loss,linestyle = '-',marker = 'x')
plt.axhline(y = -7, color = 'r', linestyle = '--',label = r"Objective at optimal solution")
#plt.title("Fine-tune Loss after Searching with Only Neighbouring CNOTs")
plt.xlabel('Epoch')
plt.ylabel('Loss (Energy)')
plt.legend()
plt.savefig('fig_qaoa_7q_1_fine_tune_loss.pdf')
plt.close()

fig = plt.figure()
plt.plot(list(range(len(qaoa_7q_2_fine_tune_loss))), qaoa_7q_2_fine_tune_loss,linestyle = '-',marker = 'x')
plt.axhline(y = -7, color = 'r', linestyle = '--',label = r"Objective at optimal solution")
#plt.title("Fine-tune Loss after Searching with Only Neighbouring CNOTs")
plt.xlabel('Epoch')
plt.ylabel('Loss (Energy)')
plt.legend()
plt.savefig('fig_qaoa_7q_2_fine_tune_loss.pdf')
plt.close()


dev_qaoa_7_qubit = qml.device('lightning.qubit', wires=7, shots=1)

def bitstring_to_int(bit_string_sample):
    bit_string = "".join(str(bs) for bs in bit_string_sample)
    return int(bit_string, base=2)
@qml.qnode(dev_qaoa_7_qubit)
def qaoa_circuit_7q_1():
    for wire in range(7):
        qml.Hadamard(wires=wire)
    qml.Rot(0.40804097338041384,
                2.145747463438778,
                0.3783766926400953, wires=2)
    qml.Rot(0.0019199100464665998,
                -1.5282123553171945,
                0.3032004878706815, wires=6)
    qml.CNOT(wires=[2,3])
    qml.Rot(1.2054773666875633e-06,
                1.5707963267944618,
                -0.15079140136872318, wires=1)
    qml.Rot(0.2932945360209,
                2.450137535981538,
                -0.11260332894569126, wires=2)
    qml.Rot(6.979993094134241e-13,
                1.5707963267950102,
                0.04144316021799416, wires=3)
    qml.Rot(-1.2971984578042746e-05,
                -1.5707963267947256,
                0.0880285652701278, wires=0)
    qml.Rot(3.0321126734711784e-13,
                -1.570796326794784,
                -0.0042588956223424046, wires=4)
    qml.Rot(-3.4354851305977896e-13,
                1.570796326791761,
                -0.7220463364203641, wires=5)
    qml.CNOT(wires=[3,2])
    qml.CNOT(wires=[2,3])

    return qml.sample()


@qml.qnode(dev_qaoa_7_qubit)
def qaoa_circuit_7q_2():
    for wire in range(7):
        qml.Hadamard(wires=wire)
    qml.Rot(-0.34052284266112803,
                -2.503167121128257,
                -0.21261523751813463, wires=6)
    qml.Rot(-1.3632510342594888e-13,
                1.5707963267952114,
                0.05700517700211684, wires=1)
    qml.CNOT(wires=[6,0])
    qml.Rot(3.8159332677590386e-13,
                -1.5707963267946412,
                0.20723775049241197, wires=4)
    qml.Rot(-1.8109147275359574e-13,
                -1.5707963267953997,
                -0.25734450068502723, wires=5)
    qml.Rot(-0.04272531861280469,
                1.3461443352298144,
                -0.19493523667279775, wires=3)
    qml.CNOT(wires=[4,5])
    qml.CNOT(wires=[5,4])
    qml.CNOT(wires=[6,5])
    qml.Rot(2.3821177174833814e-13,
                -1.570796326794622,
                0.41166743307737824, wires=0)
    qml.CNOT(wires=[5,6])
    qml.Rot(4.300116535668788e-06,
                1.5707963267954885,
                -0.032809110026752, wires=2)
    qml.Rot(-0.2029238621698908,
                -2.1672518224586232,
                -0.08096727100106925, wires=5)


    return qml.sample()


"""
VQLS Results
"""
vqls_4q_filename = "20220408-040418_VQLSDemo_4Q_QMLStateBasicGates.json"
early_stop_4q = 0.9
with open(os.path.join(cwd, vqls_4q_filename)) as f:
    vqls_4q_res_dict = json.load(f)

vqls_4q_reward_list = [c[2] for c in vqls_4q_res_dict['search_reward_list']]
vqls_4q_finetune_loss = vqls_4q_res_dict['fine_tune_loss']

fig = plt.figure()
plt.plot(list(range(len(vqls_4q_reward_list))), vqls_4q_reward_list, linestyle = '-', marker='x', label='search reward')
plt.axhline(y = early_stop_4q, color = 'r', linestyle = '--', label='early stop threshold')
plt.xlabel('Epoch')
plt.ylabel('Reward ($e^{-10C_L}$)')
plt.legend()
plt.savefig('fig_vqls_4q_search_rewards.pdf')
plt.close()

fig=plt.figure()
plt.plot(list(range(len(vqls_4q_finetune_loss))), vqls_4q_finetune_loss, linestyle = '-', marker='x', label='finetune loss')
plt.xlabel('Epoch')
plt.ylabel('Loss ($C_L$)')
plt.legend()
plt.savefig('fig_vqls_4q_finetune.pdf')
plt.close()


n_qubits = 4
sample_shots = 10**6
dev_vqls = qml.device('lightning.qubit', wires=range(n_qubits), shots=sample_shots)
J = 0.1
zeta = 1
eta = 0.2
coeff = np.array([zeta, J, J, eta])
@qml.qnode(dev_vqls)
def vqls_circ():
    for i in range(n_qubits):
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
    return qml.sample()

raw_samples = vqls_circ()
samples = []
for sam in raw_samples:
    samples.append(int("".join(str(bs) for bs in sam), base=2))

q_probs = np.bincount(samples) / sample_shots

Id = np.identity(2)
Z = np.array([[1, 0], [0, -1]])
X = np.array([[0, 1], [1, 0]])

A_0 = np.identity(4 ** 2)
A_1 = np.kron(X, np.kron(Id, np.kron(Id, Id)))
A_2 = np.kron(Id, np.kron(X, np.kron(Id, Id)))

A_3 = np.kron(Id, np.kron(Id, np.kron(Z, Z)))


A_num = coeff[0] * A_0 + coeff[1] * A_1 + coeff[2] * A_2 + coeff[3] * A_3
b = np.ones(2 ** n_qubits) / np.sqrt(2 ** n_qubits)

#print("A = \n", A_num)
#print("b = \n", b)

A_inv = np.linalg.inv(A_num)
x = np.dot(A_inv, b)

c_probs = (x / np.linalg.norm(x)) ** 2

#print("x_n^2 =\n", c_probs)
#print("|<x|n>|^2=\n", q_probs)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

ax1.bar(np.arange(0, 2 ** n_qubits), c_probs)
ax1.set_xlim(-0.5, 2 ** n_qubits - 0.5)
ax1.set_xlabel("Vector space basis")
ax1.set_title("Classical probabilities")

ax2.bar(np.arange(0, 2 ** n_qubits), q_probs)
ax2.set_xlim(-0.5, 2 ** n_qubits - 0.5)
ax2.set_xlabel("Hilbert space basis")
ax2.set_title("Quantum probabilities")

plt.savefig('fig_vqls_search_results_compare.pdf')
plt.close()








"""
Draw circuits (always at last to avoid ugly boundaries on the bar chart)
"""
qml.drawer.use_style('black_white')
fig, ax = qml.draw_mpl(qaoa_circuit_1)()
plt.savefig('fig_qaoa_circ_1.pdf')
plt.close()

fig, ax = qml.draw_mpl(qaoa_circuit_2)()
plt.savefig('fig_qaoa_circ_2.pdf')
plt.close()

fig, ax = qml.draw_mpl(qaoa_circuit_7q_1)()
plt.savefig('fig_qaoa_circ_7q_1.pdf')
plt.close()

fig, ax = qml.draw_mpl(qaoa_circuit_7q_2)()
plt.savefig('fig_qaoa_circ_7q_2.pdf')
plt.close()

fig, ax = qml.draw_mpl(vqls_circ)()
plt.savefig('fig_vqls_circ.pdf')
plt.close()

fig, ax = qml.draw_mpl(qaoa_circuit_5q_1)()
plt.savefig('fig_qaoa_5q_circ.pdf')
plt.close()
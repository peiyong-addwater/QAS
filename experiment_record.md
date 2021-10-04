# Experiment Records

## 1. Interrupted to add early stopping for search (20211004)

```python
    np.random.seed(106)
    model = ToffoliQMLSwapTestNoiseless
    marker = nowtime()
    filename = marker+'.json'
    task = model.name
    init_qubit_with_actions = {0, 1, 2}
    two_qubit_gate = ["CRot", "CNOT"]
    #single_qubit_gate = ["SX", "RZ", 'PlaceHolder']
    single_qubit_gate = ['Rot', 'PlaceHolder']
    control_map = [[0,1],[1,2],[0,2]]
    pool = QMLPool(3, single_qubit_gate, two_qubit_gate, complete_undirected_graph=False, two_qubit_gate_map=control_map)
    print(pool)
    p = 25
    l = 3
    c = len(pool)
    control_gate_limit = 5

    # set a hard limit on the number of certain gate instead of using a penalty function
    gate_limit = {"CNOT":1, "CRot":2}

    init_params = np.random.randn(p,c,l)


    final_params, final_best_arc, final_best_node, final_best_reward, final_controller, reward_list = search(
        model=model,
        op_pool=pool,
        target_circuit_depth=p,
        init_qubit_with_controls=init_qubit_with_actions,
        init_params=init_params,
        num_iterations=400,
        num_warmup_iterations=5,
        super_circ_train_optimizer=qml.AdamOptimizer,
        super_circ_train_gradient_noise_factor=0,
        super_circ_train_lr=0.1,
        penalty_function=None,
        gate_limit_dict=gate_limit,
        warmup_arc_batchsize=1000,
        search_arc_batchsize=300,
        alpha_max=2,
        alpha_min=1/np.sqrt(2)/2,
        prune_constant_max=0.95,
        prune_constant_min=0.8,
        max_visits_prune_threshold=50,
        min_num_children=3,
        sampling_execute_rounds=250,
        exploit_execute_rounds=100,
        cmab_sample_policy='local_optimal',
        cmab_exploit_policy='local_optimal',
        uct_sample_policy='local_optimal',
        verbose=2
    )

    final_params, loss_list = circuitModelTuning(
        initial_params=final_params,
        model=model,
        num_epochs=100,
        k=final_best_arc,
        op_pool=pool,
        opt_callable=qml.AdamOptimizer,
        lr=0.01,
        grad_noise_factor=0,
        verbose=1
    )
```

```

==========Model:ToffoliQMLSwapTestNoiseless, Searching at Epoch 25/400, Pool Size: 12, Arc Batch Size: 300, Search Sampling Rounds: 250, Exploiting Rounds: 100==========
     100%|██████████████████████████████| 300/300 [13:25<00:00,  2.68s/it]
Batch Training, Size = 300, Update the Parameter Pool for One Iteration
     100%|██████████████████████████████| 300/300 [00:02<00:00, 110.70it/s]
Parameters Updated!
Exploiting and finding the best arc...
Prune Count: 0
Current Best Reward: 0.9922331109623774 (After Penalization: 0.9922331109623774), Current Best Loss: 0.007766889037622571
Current Best k:
 [8, 10, 9, 0, 2, 3, 4, 3, 1, 5, 1, 1, 5, 5, 1, 3, 5, 5, 5, 1, 5, 5, 3, 1, 1]
Current Ops:
OpAtDepth: 0    OpKey: 8        OpName: {'CRot': [1, 2]}
OpAtDepth: 1    OpKey: 10       OpName: {'CRot': [0, 2]}
OpAtDepth: 2    OpKey: 9        OpName: {'CNOT': [1, 2]}
OpAtDepth: 3    OpKey: 0        OpName: {'Rot': [0]}
OpAtDepth: 4    OpKey: 2        OpName: {'Rot': [1]}
OpAtDepth: 5    OpKey: 3        OpName: {'PlaceHolder': [1]}
OpAtDepth: 6    OpKey: 4        OpName: {'Rot': [2]}
OpAtDepth: 7    OpKey: 3        OpName: {'PlaceHolder': [1]}
OpAtDepth: 8    OpKey: 1        OpName: {'PlaceHolder': [0]}
OpAtDepth: 9    OpKey: 5        OpName: {'PlaceHolder': [2]}
OpAtDepth: 10   OpKey: 1        OpName: {'PlaceHolder': [0]}
OpAtDepth: 11   OpKey: 1        OpName: {'PlaceHolder': [0]}
OpAtDepth: 12   OpKey: 5        OpName: {'PlaceHolder': [2]}
OpAtDepth: 13   OpKey: 5        OpName: {'PlaceHolder': [2]}
OpAtDepth: 14   OpKey: 1        OpName: {'PlaceHolder': [0]}
OpAtDepth: 15   OpKey: 3        OpName: {'PlaceHolder': [1]}
OpAtDepth: 16   OpKey: 5        OpName: {'PlaceHolder': [2]}
OpAtDepth: 17   OpKey: 5        OpName: {'PlaceHolder': [2]}
OpAtDepth: 18   OpKey: 5        OpName: {'PlaceHolder': [2]}
OpAtDepth: 19   OpKey: 1        OpName: {'PlaceHolder': [0]}
OpAtDepth: 20   OpKey: 5        OpName: {'PlaceHolder': [2]}
OpAtDepth: 21   OpKey: 5        OpName: {'PlaceHolder': [2]}
OpAtDepth: 22   OpKey: 3        OpName: {'PlaceHolder': [1]}
OpAtDepth: 23   OpKey: 1        OpName: {'PlaceHolder': [0]}
OpAtDepth: 24   OpKey: 1        OpName: {'PlaceHolder': [0]}

==========Epoch Time: 826.3192734718323==========


```

## 2.
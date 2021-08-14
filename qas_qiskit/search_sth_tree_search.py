from cmab_qas.search import searchParameterized
from cmab_qas.standard_ops import GatePool
from cmab_qas.circuits import (
    BitFlipSearchDensityMatrixNoiseless,
    SIMPLE_DATASET_BIT_FLIP,
    PhaseFlipDensityMatrixNoiseless,
    SIMPLE_DATASET_PHASE_FLIP,
    FourTwoTwoDetectionDensityMatrixNoiseless,
    FOUR_TWO_TWO_DETECTION_CODE_DATA,
    FiveBitCodeSearchDensityMatrixNoiseless,
    SIMPLE_DATASET_FIVE_BIT_CODE
)
import json
import numpy as np
import optax
import jax.numpy as jnp
import jax
import time

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def nowtime():
    return str(time.strftime("%Y%m%d-%H%M%S", time.localtime()))
if __name__ == "__main__":
    d_np = ["CU3Gate"]
    s_np = ["U3Gate"]
    pool = GatePool(4, s_np, d_np)
    p = 6
    l = 3
    c = len(pool)
    init_params = np.random.randn(p,c,l)
    final_params, final_best_arc, final_best_node, final_best_reward = searchParameterized(
        model=FourTwoTwoDetectionDensityMatrixNoiseless,
        data=FOUR_TWO_TWO_DETECTION_CODE_DATA,
        init_qubit_with_actions={0,1},
        init_params=init_params,
        num_iterations=500,
        num_warmup_iterations=50,
        iteration_limit=5,
        arc_batchsize=200,
        alpha=1,
        prune_constant=0.9,
        eval_expansion_factor=100,
        op_pool=pool,
        target_circuit_depth=p,
        first_policy='local_optimal',
        second_policy='local_optimal',
        super_circ_train_iterations=10,
        super_circ_train_optimizer=optax.adam,
        super_circ_train_gradient_noise_factor=1/20,
        super_circ_train_lr=0.01
    )

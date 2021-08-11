from cmab_qas.search import searchNonParameterized
from cmab_qas.standard_ops import default_complete_graph_non_parameterized_pool
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
    pool = default_complete_graph_non_parameterized_pool(4)
    searched_k, searched_node, searched_reward = searchNonParameterized(
        model=FourTwoTwoDetectionDensityMatrixNoiseless,
        data = FOUR_TWO_TWO_DETECTION_CODE_DATA,
        init_qubit_with_actions={0, 1},
        num_iterations=500,
        num_warmup_iterations=50,
        arc_batchsize=20,
        alpha=1/np.sqrt(2),
        prune_constant=0.9,
        eval_expansion_factor=100,
        op_pool=pool,
        target_circuit_depth=6,
        first_policy='local_optimal',
        second_policy='local_optimal'
    )

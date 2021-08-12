from cmab_qas.search import searchNonParameterized
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
    d_np = ["CXGate"]
    s_np = ["XGate", "YGate", "ZGate", "HGate", "TGate", "TdgGate", "SGate","SdgGate"]
    pool = GatePool(4, s_np, d_np)
    searched_k, searched_node, searched_reward = searchNonParameterized(
        model=FourTwoTwoDetectionDensityMatrixNoiseless,
        data = FOUR_TWO_TWO_DETECTION_CODE_DATA,
        init_qubit_with_actions={0, 1},
        num_iterations=1000,
        num_warmup_iterations=100,
        iteration_limit=10,
        arc_batchsize=300,
        alpha=1,
        prune_constant=0.5,
        eval_expansion_factor=100,
        op_pool=pool,
        target_circuit_depth=6,
        first_policy='local_optimal',
        second_policy='local_optimal'
    )

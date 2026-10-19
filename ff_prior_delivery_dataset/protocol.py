from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'artifacts/v7_1/world_endpoint_smoke_001'
M1 = ROOT/'artifacts/v7_1/delivery_guidance_smoke_001'
OLD = ROOT/'artifacts/v7_1/qualified_origin_learn_002'
BASE = ROOT/'artifacts/v7_1/smoke_002'
DEFAULT_OUT = ROOT/'artifacts/v7_1/delivery_dataset_128_001'
SELECTION = ROOT/'ff_prior_delivery_dataset/evidence/target_selection.json'
POPULATION = ROOT/'artifacts/v7_1/delivery_dataset_128_inputs_001/population.json'
PROTOCOL = {
    'version':'delivery_m2_fixed_TRAIN_128_1', 'targets':32, 'empty_targets':16, 'obstacle_targets':16,
    'inherited_M1_targets':8, 'additional_coverage_targets':24, 'slots_per_target':4,
    'attempts_per_slot':2, 'max_slots':128, 'max_attempts':256, 'seed':20262701,
    'target_selection':'metadata only; retain M1 8; greedy rare-origin and world-goal coverage; 12 extra per scene stratum',
    'execution_selection':'hash-order 8 targets per scene stratum, before generation; first qualified slot in frozen order; never replace execution failure',
    'execution_units':16, 'minimum_execution_verified':13, 'minimum_qualified':64, 'minimum_distinct_targets':16,
    'sampler':'M1 TRAIN-only selected method with unchanged 13774 weights',
    'candidate_wall_s':15., 'candidate_coupled_ceiling':113, 'candidate_interval_ceiling':15872,
    'DDIM_steps':16, 'repair_queries':110, 'duplicate_reference':'reject inherited or new exact reference hashes',
    'near_duplicates':'report sampled trajectory RMS; do not call slots distinct modes',
    'wall_budget_s':14400., 'cpu_threads':2, 'max_output_bytes':20*1024**3,
    'training_updates':0, 'VAL_unchanged':True, 'goal_relabel':False,
    'failed_execution_in_training':False, 'unexecuted_level':'MODEL_VALID',
    'DEV':False, 'TEST':False, 'pilot':False, 'study':False, 'strict_multimodal':False, 'old_1024':False,
    'automatic_recursive_expansion':False,
}


def proposal_id(target_slot, slot, attempt):
    if not (0<=target_slot<32 and 0<=slot<4 and 0<=attempt<2):
        raise ValueError('outside frozen attempt quota')
    return f'delivery_m2_t{target_slot:02d}_s{slot}_a{attempt}'


def proposal_seed(target_slot, slot, attempt):
    proposal_id(target_slot, slot, attempt)
    return PROTOCOL['seed']+target_slot*10007+slot*101+attempt*100003

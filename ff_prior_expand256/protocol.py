from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'artifacts/v7_1/delivery_dataset_128_001'
MODEL_RUN = ROOT/'artifacts/v7_1/delivery_train_dev_001'
MODEL_DIR = MODEL_RUN/'learning/goal_loss'
BASE = ROOT/'artifacts/v7_1/smoke_002'
DEFAULT_OUT = ROOT/'artifacts/v7_1/qualified_expand256_001'
SELECTION = ROOT/'ff_prior_expand256/evidence/target_selection.json'
POPULATION = ROOT/'artifacts/v7_1/qualified_expand256_inputs_001/population.json'
PREPARED = ROOT/'ff_prior_delivery_release/evidence/next_256_protocol.json'
PROTOCOL = {
    'version':'qualified_expand256_fixed_TRAIN_v1','targets':64,'empty_targets':32,'obstacle_targets':32,
    'slots_per_target':4,'attempts_per_slot':2,'max_slots':256,'max_attempts':512,'seed':20262901,
    'execution_units':32,'minimum_execution_verified':26,'minimum_qualified':128,'minimum_distinct_targets':32,
    'minimum_qualification_per_actual_attempt':.5,'inherited_TRAIN':13966,'VAL':48,
    'sampler':'late4_diffusion','model':'delivered13966 exact model/codec; no training or selection',
    'target_selection':'TRAIN metadata only; exclude known heldout families/tasks and previous 32 targets; greedy rare-origin/world-goal coverage; 32 per scene stratum',
    'execution_selection':'hash-order 16 targets per scene stratum before generation; first qualified frozen-order slot; no replacement after execution failure',
    'candidate_wall_s':15.,'candidate_coupled_ceiling':113,'candidate_interval_ceiling':15872,
    'DDIM_steps':16,'repair_queries':110,'duplicate_reference':'reject inherited or new exact reference hashes',
    'near_duplicates':'report trajectory RMS; slots are not distinct modes',
    'wall_budget_s':14400.,'cpu_threads':2,'max_output_bytes':20*1024**3,
    'training_updates':0,'VAL_unchanged':True,'goal_relabel':False,'failed_execution_in_training':False,
    'unexecuted_level':'MODEL_VALID','DEV':False,'TEST':False,'pilot':False,'study':False,
    'strict_multimodal':False,'old_1024':False,'automatic_recursive_expansion':False,
    'superiority_time_or_query_win_required':False,
}


def proposal_id(target_slot,slot,attempt):
    if not (0<=target_slot<64 and 0<=slot<4 and 0<=attempt<2):
        raise ValueError('outside frozen 256-slot quota')
    return f'expand256_t{target_slot:02d}_s{slot}_a{attempt}'


def proposal_seed(target_slot,slot,attempt):
    proposal_id(target_slot,slot,attempt)
    return PROTOCOL['seed'] + target_slot*100000 + slot*100 + attempt

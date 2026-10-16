from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'artifacts/v7_1/qualified_coverage_train_001'
BASE=ROOT/'artifacts/v7_1/smoke_002'
NEW_DEV=ROOT/'artifacts/v7_1/independent_new_validation_recovery_001'
DEFAULT_OUT=ROOT/'artifacts/v7_1/qualified_origin_coverage_001'
PROTOCOL={
    'version':'independent_origin_coverage_1','profile':'bounded_data_after_verified_diffusion_closure',
    'forward_TRAIN_slots':512,'forward_max_attempts':768,'attempts_per_TRAIN_slot':4,
    'new_validation_slots':8,'attempts_per_validation_slot':8,
    'TRAIN_seed':20261701,'validation_seed':20261791,'proposal_family_offset':700000,
    'origin_jitter_rad':[.04,.08,.12,.16,.20,.24],
    'motion_scales':[.5,.75,1.,1.25,1.5,1.75],
    'scene':'one scratch sphere for every fourth slot; no scene search or TRAIN parent',
    'scene_radii_m':[.012,.018,.024],'scene_offset_m':[.06,.09,.12],
    'origin_policy':'new q17 on every attempt; first qualified attempt per slot; no replacement after freeze',
    'minimum_new_validation_initial_linf_separation_rad':.02,
    'goal_guard_position_m':.01,'goal_guard_rotation_deg':1.,
    'prediction_intervals':256,'forward_coupled_ceiling':4,'candidate_wall_s':15.,
    'fixed_TRAIN_targets':8,'fixed_variants':4,'fixed_mode_admission_minimum':16,
    'fixed_mode_minimum_control_rms_rad':.005,'fixed_coupled_ceiling':113,
    'witness_TRAIN':24,'witness_new_validation':8,'maximum_native_executions':32,
    'audit_selection':'12 lowest-clearance TRAIN + 4 fixed modes + 8 hash-ordered remaining TRAIN; all 8 validation teachers; frozen before execution, no replacement',
    'wall_budget_s':5400.,'generation_reserve_for_execution_s':1200.,'validation_generation_budget_s':600.,
    'cpu_threads':2,'max_output_bytes':20*1024**3,
    'data_gate':'512 forward TRAIN with 512 unique new origins, 128 scratch scenes, >=16 fixed modes over all 8 frozen targets, 8 isolated validation tasks, all 32 native/replays and raw audit pass',
    'relative_efficiency_required':False,'training_executed_in_data_stage':False,
    'original_VAL_unchanged':48,'original_DEV_unchanged':True,'TEST':False,
    'TEST_arrays_read':False,'strict_multimodal':False,'automatic_100k_expansion':False,
    'next_stage':'freeze actual continuation training and original DEV + fresh validation comparison; do not infer training from data',
    'full_goal_auto_complete':False,
}

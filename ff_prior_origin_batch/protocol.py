from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'artifacts/v7_1/qualified_origin_train_001'
BASE=ROOT/'artifacts/v7_1/smoke_002'
DEFAULT_OUT=ROOT/'artifacts/v7_1/qualified_origin_batch_001'
PROTOCOL={
    'version':'qualified_origin_bounded_batch_1','profile':'bounded_qualified_expansion_after_C1',
    'old_TRAIN':12688,'original_VAL':48,
    'forward_TRAIN_slots':1024,'forward_admission_minimum':1000,
    'forward_max_attempts':1536,'attempts_per_TRAIN_slot':8,
    'scratch_scene_slots':256,'scratch_scene_admission_minimum':240,
    'new_validation_slots':8,'attempts_per_validation_slot':8,
    'TRAIN_seed':20262001,'validation_seed':20262091,'proposal_family_offset':900000,
    'origin_jitter_rad':[.04,.08,.12,.16,.20,.24],
    'motion_scales':[.5,.75,1.,1.25,1.5,1.75],
    'scene':'one scratch sphere for every fourth slot; no search or moved old obstacle',
    'scene_radii_m':[.012,.018,.024],'scene_offset_m':[.06,.09,.12],
    'origin_policy':'fresh q17 every attempt; first independently qualified proposal per slot',
    'minimum_new_validation_initial_linf_separation_rad':.02,
    'goal_guard_position_m':.01,'goal_guard_rotation_deg':1.,
    'prediction_intervals':256,'forward_coupled_ceiling':4,'candidate_wall_s':15.,
    'fixed_TRAIN_targets':16,'fixed_variants':4,'fixed_mode_admission_minimum':32,
    'fixed_mode_minimum_control_rms_rad':.005,'fixed_coupled_ceiling':113,
    'risk_TRAIN_audits':16,'fixed_TRAIN_audits':8,'hash_TRAIN_audits':24,
    'witness_TRAIN':48,'witness_new_validation':8,'maximum_native_executions':56,
    'audit_selection':'16 lowest-clearance TRAIN + 8 fixed modes + 24 hash-ordered remaining TRAIN; all fresh validation; freeze before execution, no replacement',
    'wall_budget_s':9000.,'generation_reserve_for_execution_s':1800.,
    'validation_generation_budget_s':600.,'cpu_threads':2,'max_output_bytes':20*1024**3,
    'data_gate':'at least1000 unique new forward origins from1024 slots, at least240 scratch scenes, >=32 fixed modes over all16 targets, fresh8 validation, all56 native/replay and raw audit, zero ERROR/unresolved',
    'parent_C1_required':True,'parent_C2_required':False,
    'relative_efficiency_required':False,'training_executed_in_data_stage':False,
    'original_DEV_unchanged':True,'used_NEW_VAL8_role_in_this_protocol':'DEV_guard_only; original source remains unchanged',
    'TEST':False,'TEST_arrays_read':False,'strict_multimodal':False,
    'automatic_100k_expansion':False,'full_goal_auto_complete':False,
    'next_stage':'separately freeze actual training of every qualified new row and independent validation; do not infer neural benefit from data',
}

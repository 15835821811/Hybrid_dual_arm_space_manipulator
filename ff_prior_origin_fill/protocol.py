from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'artifacts/v7_1/qualified_origin_coverage_001'
DEFAULT_OUT=ROOT/'artifacts/v7_1/qualified_origin_fill_001'
DIAGNOSTIC=ROOT/'ff_prior_pilot_tools/evidence/origin_contact_diagnostic_002/diagnostic.json'
SLOTS=(171,211,243,271,279,351,431,443,491)
PROTOCOL={'version':'bounded_origin_fill_and_merged_release_1','profile':'qualified_data_supplement',
    'source_original_gate_result':False,'preserve_original_gate_and_errors':True,'source_status_preserved':'FINISHED_WITH_BLOCKED_ITEMS',
    'slots':list(SLOTS),'attempts_per_slot':8,'max_attempts':72,'seed':20261801,
    'proposal_family_offset':800000,'origin_jitter_rad':[.04,.08,.12,.16,.20,.24],
    'motion_scales':[.5,.75,1.,1.25,1.5,1.75],
    'scene_radii_m':[.012,.018,.024],'scene_offset_m':[.06,.09,.12],
    'scene':'one new scratch sphere per attempt; no old scene or target relabelling',
    'goal_prediction_intervals':256,'forward_coupled_ceiling':4,'candidate_wall_s':15.,
    'minimum_validation_initial_linf_separation_rad':.02,
    'goal_guard_position_m':.01,'goal_guard_rotation_deg':1.,
    'selection':'first model-qualified new task per source coverage slot; freeze all before native; no replacement',
    'maximum_new_executions':9,'inherited_executions':32,'wall_budget_s':1800.,
    'generation_reserve_for_audit_s':600.,'cpu_threads':2,'max_output_bytes':10*1024**3,
    'merged_forward':512,'merged_obstacle_forward':128,'merged_fixed_modes':32,
    'merged_TRAIN':12688,'original_VAL':48,'frozen_NEW_VAL':8,
    'merged_gate':'all included records independently admitted; 512 distinct new forward origins and 128 scratch scenes; 32 fixed modes over 8 frozen targets; unchanged validation; inherited32 and new9 native/replays; raw/provenance audit PASS; no new ERROR or unresolved work',
    'historical_errors_and_costs_retained':True,'training_executed':False,'TEST':False,
    'TEST_arrays_read':False,'strict_multimodal':False,'automatic_100k_expansion':False,
    'relative_efficiency_required':False,'full_goal_auto_complete':False,
    'next_stage':'separately frozen actual training of merged data and original DEV/fresh-validation comparison'}

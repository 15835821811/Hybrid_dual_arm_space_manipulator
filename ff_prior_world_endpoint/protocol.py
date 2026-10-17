from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'artifacts/v7_1/qualified_origin_learn_002'
BASE=ROOT/'artifacts/v7_1/smoke_002'
DEFAULT_OUT=ROOT/'artifacts/v7_1/world_endpoint_smoke_001'
METHODS=['structured','frozen_diffusion','learned_mean','expanded_diffusion']
PROTOCOL={
 'version':'nonlinear_world_endpoint_supervision_1','old_TRAIN':13774,'original_VAL':48,
 'fixed_TRAIN_targets':16,'fixed_variants':4,'minimum_modes':32,'maximum_modes':64,
 'TRAIN_native_witnesses':16,'target_selection':'hash-ordered8 empty +8 sphere from qualified_origin_batch_001 TRAIN; frozen before correction',
 'mode_minimum_control_rms_rad':.005,'mode_coupled_ceiling':113,
 'training_updates':6000,'batch':64,'training_seed':20262301,'lr':2e-4,
 'world_weight':.1,'world_every_updates':50,'world_updates':120,
 'world_noise_times':[500,999],'world_noise_seed':20262302,
 'endpoint_gradient_dimensions':102,'endpoint_difference_normalized':.01,
 'endpoint_intervals':256,'endpoint_convergence_intervals':512,
 'endpoint_gradient_half_tolerance':.02,'endpoint_gradient_direction_tolerance':.05,
 'queries_per_gradient':212,'world_gradient_query_ceiling':25440,
 'diagnostic_noise_seed':20262303,'TRAIN_endpoint_diagnostic_cases':32,
 'endpoint_loss':'sum of both arms squared position/5mm and geodesic rotation/1degree, divided by4; actual predicted raw trajectory',
 'gradient':'full102 two-sided finite differences at current model prediction; no teacher-local surrogate',
 'sampling_seeds':[20262119,20262123,20262129],'methods':METHODS,
 'original_DEV_tasks':4,'reused_development_tasks':8,'independent_new_validation':False,
 'candidates_per_task':5,'max_candidates':720,'max_selected_executions':144,'max_diagnostics':96,
 'candidate_wall_s':15.,'candidate_coupled_ceiling':113,'candidate_interval_ceiling':15872,
 'DDIM_steps':16,'selection':'minimum normalized frozen goal error among eligible, one execution, no replacement',
 'stability':'each seed original DEV>=18/20 and reused DEV8>=36/40; new all36 execution/replay; zero errors',
 'quality':'task-paired pre-correction median ratio<=.85 and bootstrap geometric95upper<1 vs frozen13774; report strong baselines separately',
 'bootstrap_samples':10000,'bootstrap_seed':20262171,
 'wall_budget_s':14400.,'training_reserve_for_smoke_s':4200.,'diagnostic_reserve_s':600.,
 'cpu_threads':2,'max_output_bytes':20*1024**3,
 'runtime_cache_entries':64,'runtime_scene_models':4,
 'max_process_private_bytes':16*1024**3,'minimum_available_memory_bytes':2*1024**3,
 'TEST':False,'TEST_arrays_read':False,'strict_multimodal':False,'broad1024_expansion':False,
 'relative_efficiency_required':False,'full_goal_auto_complete':False,
 'conditional_TEST_gate':'after raw audit and independent review: C1 stable, TRAIN paired true endpoint median ratio<=.8, quality vs frozen13774 passes, fixed raw>=4/12 goals and12/12 physics; freeze one untouched TEST protocol before accessing TEST',
 'TEST_automatically_launched_by_this_worker':False,
}

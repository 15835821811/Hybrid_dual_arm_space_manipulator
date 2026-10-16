"""Frozen before task generation or any competing method is run."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'artifacts/v7_1/qualified_coverage_train_001'
OLD = ROOT / 'artifacts/v7_1/qualified_pilot_train_002'
BASE = ROOT / 'artifacts/v7_1/smoke_002'
METHODS = ('structured', 'frozen_diffusion', 'learned_mean', 'expanded_diffusion')
GENERATION = {
    'seed': 20261601, 'slots': 12, 'attempts_per_slot': 8,
    'origin_jitter_rad': [.14, .16, .18, .20, .22, .24],
    'motion_scales': [1.25, 1.5, 1.75], 'obstacle_slots': list(range(6, 12)),
    'obstacle_bow_rad': .12, 'proposal_family_offset': 600000,
    'selection': 'first independently qualified teacher per fixed slot; before planners',
    'scene': 'new scratch static sphere selected by frozen teacher/public-direct screen',
    'scene_parent': None, 'reuse_TRAIN_teacher': False,
    'goal_guard_position_m': .01, 'goal_guard_rotation_deg': 1.,
    'attempt_wall_s': 45., 'generation_budget_s': 1800.,
    'TEST_metadata_guard_only': True, 'TEST_arrays_read': False,
}
PROTOCOL = {
    'version': 'independent_world_goal_validation_1', 'profile': 'bounded_development_validation',
    'generation': GENERATION, 'methods': list(METHODS), 'task_slots': 12,
    'sampling_seeds': [20261619, 20261623, 20261629], 'candidates_per_task': 5,
    'max_candidates': 720, 'max_selected_executions': 144, 'max_witness_executions': 12,
    'candidate_wall_s': 15., 'candidate_coupled_ceiling': 113,
    'candidate_interval_ceiling': 15872, 'ddim_steps': 16,
    'wall_budget_s': 7200., 'cpu_threads': 2, 'max_output_bytes': 20 * 1024**3,
    'raw_diagnostics': {'seed_group': 0, 'candidate_id': 0,
                        'stages': ['raw', 'smoothed_clipped'], 'maximum': 96,
                        'separate_budget_s': 600., 'affects_selection': False},
    'selection': 'minimum normalized frozen-goal error among eligible; one execution; no replacement',
    'stability': '12 frozen tasks; 12/12 teacher execution/replay; new model each seed >=54/60 eligible; all 36 selected execution/replay; no numerical ERROR',
    'quality_claim': 'task median pre-correction pose norm ratio <=0.85 and paired task-bootstrap 95% upper ratio <1 against BOTH structured and learned_mean, with stability and no physical success regression',
    'bootstrap_seed': 20261691, 'bootstrap_replicates': 10000,
    'training_updates': 0, 'checkpoint_selection': False, 'obstacle_conditioned': False,
    'original_DEV_modified': False, 'TEST': False, 'strict_multimodal': False,
    'relative_efficiency_required': False, 'automatic_100k_expansion': False,
    'full_goal_auto_complete': False,
}

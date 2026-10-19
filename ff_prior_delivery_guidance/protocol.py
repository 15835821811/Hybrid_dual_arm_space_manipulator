from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'artifacts/v7_1/world_endpoint_smoke_001'
OLD = ROOT / 'artifacts/v7_1/qualified_origin_learn_002'
BASE = ROOT / 'artifacts/v7_1/smoke_002'
DEFAULT_OUT = ROOT / 'artifacts/v7_1/delivery_guidance_smoke_001'
METHODS = ('structured', 'learned_mean', 'frozen_diffusion', 'late4_diffusion')
ALLOCATIONS = {12: 27, 13: 27, 14: 27, 15: 29}
PROTOCOL = {
    'version': 'delivery_m1_late_clean_guidance_1',
    'model_sha256': '02b2af4dfa4ccef027fdebd246c92635c9c97f338c5c5181075176fd380994ff',
    'TRAIN_targets': 8, 'target_selection': 'SHA256(sample_id) order; 4 empty + 4 obstacle endpoint_mode_00..15_v0',
    'sampling_seeds': [20262601, 20262602], 'methods': list(METHODS),
    'candidates_per_task': 5, 'max_candidates': 320, 'max_selected_executions': 64,
    'DDIM_steps': 16, 'prediction': 'v', 'guidance_allocations': {str(k): v for k, v in ALLOCATIONS.items()},
    'repair_queries': 110, 'candidate_coupled_ceiling': 113, 'candidate_interval_ceiling': 15872,
    'candidate_wall_s': 15., 'diagnostic_validations': 16, 'diagnostic_coupled_ceiling': 48,
    'diagnostic_gate': 'original teachers must qualify; if any latent roundtrip is not eligible, stop late4 only and retain baseline comparison',
    'selection': 'min(score,candidate_id) among eligible; one native execution, no replacement',
    'method_selection': 'TRAIN only: late4 must pass engineering gate and both counts >= frozen_diffusion; else old validated sampler',
    'minimum_eligible_per_method': 64, 'minimum_execution_verified_per_method': 13,
    'wall_budget_s': 14400., 'cpu_threads': 2, 'max_output_bytes': 20 * 1024**3,
    'training_updates': 0, 'DEV': False, 'TEST': False, 'pilot': False,
    'old_1024': False, 'strict_multimodal': False, 'goal_relabel': False,
    'relative_efficiency_required': False, 'advantage_required_for_delivery': False,
}

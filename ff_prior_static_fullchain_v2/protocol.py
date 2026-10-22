from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'ff_prior_static_fullchain_v2'
OUT = ROOT / 'artifacts/v7_1/static_fullchain_002'
SOURCE = ROOT / 'artifacts/v7_1/sample_alignment_001'
BASE = ROOT / 'artifacts/v7_1/smoke_002'
CODEC = ROOT / 'artifacts/v7_1/control_transfer_bounded_001/codec.json'
P = {
    'version': 'public_static_fullchain_2_short_paths',
    'tasks': ['capture_3_obstacles', 'capture_5_obstacles'],
    'seeds': [20264611, 20264617, 20264623],
    'methods': ['structured_geometry', 'diffusion_geometry'],
    'proposals_per_unit': 5, 'selected_repairs': 2, 'repair_iterations': 16,
    'execution_units': 12, 'candidate_denominator': 60,
    'network_steps': 16, 'detail_initialization_seed': 20264601,
    'bow_margin_m': .06, 'bow_control_cap_rad': .3,
    'screen_intervals': 512, 'raw_intervals': 256,
    'screen_rank': 'collision_violation + .05 * squared_normalized_pose; then candidate id',
    'sparse_base_grids': [512, 1024, 2048], 'active_pair_cap': 240,
    'active_buffer_m': .004, 'dense_grids': [512, 1024, 2048, 4096, 8192, 16384],
    'validation_intervals': [256, 512, 1024, 2048, 4096, 8192, 16384],
    'difference_rad': .0001, 'trust_rad': .15, 'trust_initial_rad': .0375,
    'trust_min_rad': .001171875, 'ratio_accept': .1, 'ratio_shrink': .25,
    'ratio_expand': .75, 'restore_bisections': 50, 'solver_max_iterations': 160,
    'line_scales': [1., .5, .25, .125, .0625, .03125],
    'repair_clearance_m': .0002, 'collision_scale_m': .003,
    'sparse_agreement_m': .001, 'sparse_agreement_rad': .003490658503988659,
    'coupled_cap': 147000, 'integration_cap': 352000000,
    'unit_coupled_cap': 12000, 'unit_integration_cap': 29000000,
    'repair_coupled_cap': 5800, 'repair_integration_cap': 14000000,
    'unit_wall_s': 1650, 'repair_wall_s': 600, 'execution_reserve_s': 360,
    'wall_s': 21560, 'cpu_threads': 2, 'output_bytes_cap': 20*1024**3 - 128*1024**2,
    'native_steps_cap': 360000, 'TRAIN_added': 0, 'training_updates': 0,
    'TEST': False, 'expand': False, 'physics_relaxed': False, 'goal_relabel': False,
    'qualification': 'original nonpenetration, dual world 5mm/1deg, 2mm geometry, full motion/native67/replay gates',
    'claim_scope': 'development engineering feasibility; diffusion benefit separately judged',
}

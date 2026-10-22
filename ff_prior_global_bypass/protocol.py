from pathlib import Path
from ff_prior_static_fullchain_v2.protocol import P as ORIGINAL

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = Path(__file__).resolve().parent
OUT = ROOT/'artifacts/v7_1/global_bypass_001'
SOURCE = ROOT/'artifacts/v7_1/sample_alignment_001'
BASE = ROOT/'artifacts/v7_1/smoke_002'
CODEC = ROOT/'artifacts/v7_1/control_transfer_bounded_001/codec.json'
P = dict(ORIGINAL, version='forward_floating_state_rrt_quintic_to323_v1',
    tasks=['capture_5_obstacles'], candidate_denominator=30, execution_units=6,
    coupled_cap=36000, integration_cap=80000000, unit_coupled_cap=5990,
    unit_integration_cap=13200000, unit_wall_s=3500, wall_s=21540,
    output_bytes_cap=20*1024**3-128*1024**2, native_steps_cap=165000,
    seed_coupled_cap=220, seed_integration_cap=150000,
    search_coupled_cap=500, search_integration_cap=900000,
    candidate_wall_s=410, search_attempt_cap=400, accepted_node_cap=400,
    edge_grids=[32,64,128,256,512], edge_max_step_m=.002,
    edge_max_joint_step_rad=.18, edge_margin_m=.0002,
    tree_goal_probability=.25, tree_proposal_probability=.30,
    tree_uniform_probability=.15, tree_sample_continuum_scale=.10,
    tree_sample_rigid_scale=.65, tree_goal_connection_period=32,
    tree_endpoint_iterations=6, tree_endpoint_step_rad=.12,
    tree_endpoint_difference_rad=.0001, tree_goal_reserve_s=4.,
    tree_min_duration_s=.35, tree_time_margin=1.10,
    fit_samples=513, fit_endpoint_weight=10., selected_repairs=1,
    repair_iterations=6, repair_coupled_cap=2100,
    repair_integration_cap=6200000, repair_wall_s=950,
    validation_coupled_cap=40, validation_integration_cap=200000,
    execution_reserve_s=360, mechanism_attempts=1,
    screen_rank='certified eligible first; then original goal+geometry merit; then candidate id',
    full_integration_replay_required=True)

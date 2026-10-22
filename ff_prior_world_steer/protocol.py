from pathlib import Path
from ff_prior_static_fullchain_v2.protocol import P as ORIGINAL

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = Path(__file__).resolve().parent
OUT = ROOT/'artifacts/v7_1/world_steer_001'
SOURCE = ROOT/'artifacts/v7_1/sample_alignment_001'
BASE = ROOT/'artifacts/v7_1/smoke_002'
CODEC = ROOT/'artifacts/v7_1/control_transfer_bounded_001/codec.json'
P = dict(ORIGINAL, version='reusable_world_goal_short_steer_to323_v1',
    tasks=['capture_5_obstacles'], methods=['diffusion_geometry'],
    candidate_denominator=15, execution_units=3,
    coupled_cap=60000, integration_cap=110000000, unit_coupled_cap=19980,
    unit_integration_cap=35000000, unit_wall_s=6800, wall_s=21540,
    output_bytes_cap=20*1024**3-128*1024**2, native_steps_cap=82500,
    d3_cycle_native_cap=165000, remaining_dynamic_native_reserve=82500,
    seed_coupled_cap=220, seed_integration_cap=150000,
    search_coupled_cap=3000, search_integration_cap=5000000,
    candidate_wall_s=950, search_attempt_cap=400, accepted_node_cap=400,
    edge_grids=[32,64,128,256,512,1024], edge_max_step_m=.002,
    edge_max_joint_step_rad=.18, edge_margin_m=.0002,
    tree_goal_probability=.25, tree_proposal_probability=.30,
    tree_uniform_probability=.15, tree_sample_continuum_scale=.10,
    tree_sample_rigid_scale=.65, tree_goal_connection_period=32,
    tree_endpoint_iterations=6, tree_endpoint_step_rad=.12,
    tree_endpoint_difference_rad=.0001, tree_goal_reserve_s=4.,
    tree_min_duration_s=.35, tree_time_margin=1.10,
    steer_duration_s=1., steer_difference_intervals=256, steer_damping=.001,
    steer_max_step_rad=.12, steer_parent_topk=4,
    steer_parent_visit_weight=.25, steer_parent_time_weight=.04,
    world_nearest_weight=.15, steer_terminal_margin=.7,
    steer_query_reserve=48, steer_interval_reserve=14000,
    steer_world_probability=.60, steer_proposal_probability=.25,
    steer_line_scales=[1.,.5,.25], steer_max_depth=22,
    fit_samples=513, fit_endpoint_weight=10., selected_repairs=1,
    repair_iterations=8, repair_coupled_cap=3500,
    repair_integration_cap=8000000, repair_wall_s=1500,
    validation_coupled_cap=40, validation_integration_cap=200000,
    execution_reserve_s=360, mechanism_attempts=1,
    screen_rank='certified eligible first; then original goal+geometry merit; then candidate id',
    full_integration_replay_required=True, advantage_comparison=False,
    render_frames=300,render_width=960,render_height=720,render_fps=15,visualization_wall_s=600,
    claim_scope='single-system development engineering; no relative-advantage experiment')

from pathlib import Path
from ff_prior_static_fullchain_v2.protocol import P as ORIGINAL
ROOT=Path(__file__).resolve().parents[1]
PACKAGE=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/v7_1/static_goal_seed_001'
SOURCE=ROOT/'artifacts/v7_1/sample_alignment_001'
BASE=ROOT/'artifacts/v7_1/smoke_002'
CODEC=ROOT/'artifacts/v7_1/control_transfer_bounded_001/codec.json'
P=dict(ORIGINAL,version='public_nonlinear_goal_seed_then_full323',
    tasks=['capture_5_obstacles'],candidate_denominator=30,execution_units=6,
    selected_repairs=1,repair_iterations=24,goal_iterations=18,
    goal_intervals=512,goal_target_Q=.5,goal_difference_rad=1e-4,
    goal_damping=1e-3,goal_step_rad=.15,goal_coupled_cap=400,
    goal_integration_cap=250000,goal_wall_s=100,
    coupled_cap=64000,integration_cap=290000000,
    unit_coupled_cap=10600,unit_integration_cap=48000000,
    repair_coupled_cap=8200,repair_integration_cap=46000000,
    unit_wall_s=2200,repair_wall_s=1100,execution_reserve_s=360,
    wall_s=13800,output_bytes_cap=15*1024**3,native_steps_cap=165000,
    screen_rank='collision_violation + .05 * squared_normalized_pose after nonlinear goal seed; then candidate id',
    mechanism_attempts=1,P1_prior_reserved_coupled=95434,
    P1_remaining_coupled=64566,P1_remaining_wall_s=14021.469,
    P1_remaining_integrations=297980076,
    P1_prior_reserved_bytes=3419458053+128*1024**2,
    full_integration_replay_required=True)

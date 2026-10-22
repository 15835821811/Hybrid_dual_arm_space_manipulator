from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PACKAGE=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/v7_1/fullchain_acceptance_001'
STATIC=ROOT/'artifacts/v7_1/static_fullchain_002'
DYNAMIC=ROOT/'artifacts/v7_1/temporal_learning_001'
FIVE=ROOT/'artifacts/v7_1/static_goal_seed_001'
BASE=ROOT/'artifacts/v7_1/smoke_002'
P=dict(version='fixed_development_acceptance_v1',threads=2,wall_s=21600,
    output_bytes_cap=20*1024**3,new_native_replay_steps=165000,
    new_static_units=6,logical_units=30,per_unit_wall_s=1800,
    new_training_updates=0,new_proposals=0,TEST=False,TRAIN_added=0,
    methods=['deterministic_system','diffusion_system'],
    scenes=['capture_3_obstacles','capture_5_obstacles','stationary','drifting','five_obstacle_to_drifting'],
    exact_old_evidence_only=True,full_integration_atol=1e-10,
    no_replacement=True,no_same_bottleneck_retry=True,geometry_step_m=.002)

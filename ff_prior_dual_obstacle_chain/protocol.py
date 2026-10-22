from pathlib import Path
from ff_prior_diffusion_demo.protocol import P as STATIC
ROOT=Path(__file__).resolve().parents[1]
PACKAGE=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/v7_1/dual_obstacle_chain_001'
SOURCE=ROOT/'artifacts/v7_1/sample_alignment_001'
BASE=ROOT/'artifacts/v7_1/smoke_002'
DYNAMIC=ROOT/'artifacts/v7_1/dynamic_handoff_001/u00/task.json'
P=dict(STATIC,version='dual_arm_obstacles_fixed_seven_waypoints_v1',
    wall_s=10800,unit_wall_s=2400,coupled_cap=7000,unit_coupled_cap=7000,
    integration_cap=17000000,unit_integration_cap=17000000,
    repair_coupled_cap=3450,repair_integration_cap=8300000,repair_wall_s=900,
    native_steps_cap=54500,output_bytes_cap=12*1024**3-128*1024**2,
    max_bytes=12*1024**3,steps_each=13500,max_native_replay_steps=27000,
    dt=.002,seed=20261020,continuum_obstacles=6,obstacle_radius_m=.012,
    bypass_offset_m=[-.06,0.,0.],dynamic_duration_s=27.,
    task_split='APPLICATION_DEV_DUAL_OBSTACLE',training=False,expansion=False,
    executions=2,comparison_runs=0,TEST=False)

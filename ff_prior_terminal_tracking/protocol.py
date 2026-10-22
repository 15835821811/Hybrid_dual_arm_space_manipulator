from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PACKAGE=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/v7_1/terminal_tracking_001'
SOURCE=ROOT/'artifacts/v7_1/diffusion_demo_001'
DYNAMIC=ROOT/'artifacts/v7_1/dynamic_handoff_001/u00/task.json'
P=dict(version='actual_terminal_tracking_v1',duration=27.,dt=.002,steps_each=13500,
       threads=2,wall_s=14400,max_bytes=12*1024**3,max_native_replay_steps=27000,
       max_coupled=0,executions=1,seed=20261020,training=False,expansion=False,TEST=False,
       preserve_world_waypoints=True,preserve_static_goal=True,actual_target_velocity_unchanged=True,
       position_gate_m=.005,rotation_gate_deg=1.,geometry_step_m=.002,replay_atol=1e-10)

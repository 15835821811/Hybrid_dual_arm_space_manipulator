from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PACKAGE=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/v7_1/dynamic_handoff_001'
OLD=ROOT/'artifacts/v7_1/waypoint_domain_002'
MODEL=ROOT/'artifacts/v7_1/endpoint_jacobian_bounded_001/learning'
GATE=ROOT/'artifacts/v7_1/dynamic_margin_001'
P=dict(stage='P2b_original_dynamic_handoff',duration=27.,dt=.002,steps_each=13500,
       threads=1,wall_s=14400,max_bytes=12*1024**3,max_native_replay_steps=297000,
       planned_unique_executions=8,logical_units=12,ddim_steps=16,
       task_seeds={'stationary':20261020,'drifting':20261021},
       proposal_seeds={'stationary':[20261020,20262020,20263020],
                       'drifting':[20261021,20262021,20263021]},
       training=False,expansion=False,TEST=False)

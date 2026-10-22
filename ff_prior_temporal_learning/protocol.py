from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PACKAGE=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/v7_1/temporal_learning_001'
DATA=ROOT/'artifacts/v7_1/temporal_data_001'
DEV=ROOT/'artifacts/v7_1/dynamic_handoff_001'
P=dict(stage='P3b_full_temporal_learning_and_fixed_development',seed=20261022,
       updates=3000,batch=8,lr=2e-4,weight_decay=.01,clip_grad=1.,threads=1,
       channels=48,dilations=[1,2,4,8],ddim_steps=16,diagnostic_updates=32,
       rollout_every=4,rollout_batch=2,rollout_weight=1.,detail_weight=.1,
       train_rows=8,val_rows=4,methods=['deterministic_sequence','temporal_diffusion'],
       duration=27.,dt=.002,steps_each=13500,free_coefficients=[271,17],
       max_obstacles=5,world_feedback_tracking_gain=.5,world_feedback_regularizer=1e-4,
       max_native_replay_steps=432000,wall_s=18000,max_bytes=16*1024**3,
       training_wall_s=10800,TEST=False,expansion=False,validation_in_gradient=False)

from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PACKAGE=ROOT/'ff_prior_full_control'
OUT=ROOT/'artifacts/v7_1/full_control_bounded_001'
SOURCE=ROOT/'artifacts/v7_1/sample_alignment_001'
DATA=ROOT/'artifacts/v7_1/expand256_train_dev_001'
BASE=ROOT/'artifacts/v7_1/smoke_002'
P={'version':'full323_preserve_full16_teacher_1','TRAIN':14222,'VAL':48,
   'rank':19,'dim':323,'condition_dim':65,'fit':'TRAIN_only','ridge_lambda':10.,
   'updates':6000,'batch':64,'lr':2e-4,'weight_decay':.01,'training_seed':20264101,
   'initialization':'fresh seeded MLP; not functional warm start from 102 dimensions',
   'rollout_every':5,'rollout_batch':8,'rollout_seed':20264111,'rollout_norm_cap':.5,
   'loss':'v MSE + capped full16 final control teacher loss; no claimed nonlinear world loss',
   'audit_updates':[1,5,3000,6000],'ddim_steps':16,
   'val_seeds':[20263111,20263113],'sampling_seeds':[20262119,20262123,20262129],
   'reconstruction_cases':6,'reconstruction_executions':3,'reconstruction_control_error_rad':1e-6,
   'reconstruction_gate':'float32 training-target reconstruction; 6/6 MODEL_VALID and 3/3 native/replay; no retained updates',
   'validation_intervals':[256,512,1024,2048,4096,8192,16384],
   'candidate_coupled_cap':113,'candidate_wall_s':15.,'candidate_intervals_cap':15872,
   'new_DEV_candidates':180,'new_DEV_execution_units':36,'new_application_candidates':10,'new_application_execution_units':2,
   'coupled_cap':30000,'wall_s':21600,'cpu_threads':2,'output_bytes_cap':20*1024**3,
   'TRAIN_added':0,'TEST':False,'expand':False,'physics_relaxed':False,'goal_relabel':False,
   'qualification':'unchanged only-nonpenetration, 5mm/1deg, 2mm geometry and all original gates'}

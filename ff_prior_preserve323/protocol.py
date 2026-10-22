from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PACKAGE=ROOT/'ff_prior_preserve323'
OUT=ROOT/'artifacts/v7_1/preserve323_bounded_001'
SOURCE=ROOT/'artifacts/v7_1/sample_alignment_001'
DATA=ROOT/'artifacts/v7_1/expand256_train_dev_001'
BASE=ROOT/'artifacts/v7_1/smoke_002'
REUSE=ROOT/'artifacts/v7_1/control_transfer_bounded_001'
P={'version':'functional_low102_full323_detail_1','TRAIN':14222,'VAL':48,
   'rank':19,'dim':323,'low_dim':102,'condition_dim':65,'fit':'TRAIN_only',
   'updates':6000,'batch':64,'lr':1e-4,'weight_decay':.01,'training_seed':20264201,
   'initialization':'exact old aligned low102 weights and codec; zero clean detail head',
   'rollout_every':5,'rollout_batch':8,'rollout_seed':20264211,'rollout_norm_cap':.5,
   'detail_bound_sigma':3.,'detail_weight':.05,'temporal_error_weight':.25,
   'loss':'low v + detail clean MSE; capped full16 physical teacher and second difference error; no nonlinear world gradient',
   'audit_updates':[1,5,3000,6000],'ddim_steps':16,'detail_noise_offset':971003,
   'diagnostic_updates':256,'diagnostic_lr':1e-4,'diagnostic_targets':8,'diagnostic_seeds':[20264231,20264237],
   'migration_error_rad':1e-12,'learnability_loss_ratio':.9,'learnability_qualified_min':12,
   'learnability_q_ratio':.9,'learnability_roughness_ratio':1.10,
   'val_seeds':[20263111,20263113],'sampling_seeds':[20262119,20262123,20262129],
   'reconstruction_cases':6,'reconstruction_executions':3,'reconstruction_control_error_rad':1e-6,
   'validation_intervals':[256,512,1024,2048,4096,8192,16384],
   'candidate_coupled_cap':113,'candidate_wall_s':15.,'candidate_intervals_cap':15872,
   'new_DEV_candidates':180,'new_DEV_execution_units':36,'new_application_candidates':10,'new_application_execution_units':2,
   'coupled_cap':33000,'wall_s':21600,'cpu_threads':2,'output_bytes_cap':20*1024**3,
   'TRAIN_added':0,'TEST':False,'expand':False,'physics_relaxed':False,'goal_relabel':False,
   'qualification':'unchanged only-nonpenetration, 5mm/1deg, 2mm geometry and original motion/native67/replay gates'}
P.update(version='full16_world_residual_trust_and_qualified_protection_1',
    world_rounds=4, world_inner_updates=64, world_lr=1e-4,
    world_teacher_weight=.05, world_violation_weight=1., world_protection_weight=100.,
    world_trust_weight=.01, world_protection_margin=.9,
    world_difference=.01, world_gradient_dimensions=323, world_queries_per_jacobian=654,
    world_half_tolerance=.02, world_direction_tolerance=.05,
    world_intervals=256, world_fine_intervals=512,
    world_trial_scales=[1.,.5,.25], world_accept_mean_ratio=.99,
    world_query_cap=112000, coupled_cap=112000,
    loss='ordinary transfer323 objective then bounded refreshed full323 world residual Jacobian through full16 DDIM; real-physics line search protects prior qualification',
    formal_candidate_initialization='original exact aligned transfer, never M1 diagnostic weights',
    formal_refinement='after 6000 updates, at most four fresh-Jacobian rounds; keep formal final weights, no model selection on VAL/DEV',
    diagnostic_updates=256)

PREVIOUS=ROOT/'artifacts/v7_1/rollout_world_bounded_001'
P.update(version='frozen_low102_stratified_protection_heldout_TRAIN_1', training_seed=20265001,
    selection_seed=20265011, diagnostic_seeds=[20265021,20265023], world_rounds=3,
    world_replay_weight=5., fit_targets=16, check_targets=16, fit_outputs=32, check_outputs=32,
    check_median_ratio=1.05, check_p95_ratio=1.05, qualified_fit_min=16,
    protected_min_each_pool=2, ordinary_protection_weight=5., coupled_cap=160000,
    frozen_low=True, check_pool_gradient=False, codec_reused=True,
    loss='frozen low102; train detail221 with teacher and fit-protected replay; refreshed full16 world supervision with fit-only acceptance and one independent TRAIN check',
    formal_candidate_initialization='original aligned low102 with zero detail; never diagnostic weights')

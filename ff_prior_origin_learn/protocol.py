from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'artifacts/v7_1/qualified_origin_batch_001'
OLD=ROOT/'artifacts/v7_1/qualified_origin_train_001'
BASE=ROOT/'artifacts/v7_1/smoke_002'
DEFAULT_OUT=ROOT/'artifacts/v7_1/qualified_origin_learn_001'
PRODUCERS=('qualified_origin_batch_001',)
METHODS=('structured','frozen_diffusion','learned_mean','expanded_diffusion')
PROTOCOL={
    'version':'qualified_origin_recalibrated_training_and_smoke_1',
    'TRAIN':13774,'old_TRAIN':12688,'new_coverage_TRAIN':1086,'VAL':48,
    'training_updates':6000,'batch':64,'uniform_rows_per_batch':32,'uniform_origins_per_batch':32,'TRAIN_origins':1578,
    'training_seed':20262101,'lr':2e-4,'local_coupled_pose_loss_weight':.05,
    'warm_start':'verified 12688 model; fixed temporal basis and latent normalization; TRAIN-only affine condition recalibration',
    'mean_baseline':'TRAIN-only row/origin mixture weighted ridge with fixed temporal basis, lambda10',
    'sampling_seeds':[20262119,20262123,20262129],
    'original_DEV_tasks':4,'independent_NEW_VAL_tasks':8,'candidates_per_task':5,
    'methods':list(METHODS),'max_candidates':720,'max_selected_executions':144,
    'diagnostic':'first seed candidate0, raw and common smoothed/clipped; after all selection',
    'max_diagnostics':96,'candidate_wall_s':15.,'candidate_coupled_ceiling':113,
    'candidate_interval_ceiling':15872,'DDIM_steps':16,
    'guidance':'common hard target-preserving projector at final clean denoising step',
    'selection':'minimum normalized frozen goal error among eligible, one execution, no replacement',
    'stability':'new each seed DEV>=18/20 and NEW_VAL>=36/40, all 36 selected execution/replay, zero candidate numerical errors',
    'quality':'NEW_VAL8 task-paired pre-correction median norm ratio<=.85 and bootstrap geometric95 upper<1 versus both structured and expanded weighted mean; stable and no eligible/native regression on NEW_VAL',
    'bootstrap_seed':20262171,'bootstrap_samples':10000,
    'wall_budget_s':14400.,'training_reserve_for_smoke_s':4200.,'diagnostic_reserve_s':600.,
    'cpu_threads':2,'max_output_bytes':20*1024**3,
    'NEW_VAL_in_training_or_codec_fit_or_checkpoint_selection':False,
    'VAL_gradients':False,'checkpoint_selection':'fixed final update',
    'data_size_causality_claim':False,'relative_efficiency_required':False,
    'TEST':False,'TEST_arrays_read':False,'strict_multimodal':False,
    'automatic_100k_expansion':False,'full_goal_auto_complete':False,
}

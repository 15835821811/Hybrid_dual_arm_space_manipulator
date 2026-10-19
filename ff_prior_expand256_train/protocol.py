from pathlib import Path
from ff_prior.storage import read_json
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'artifacts/v7_1/qualified_expand256_001'
OLD=ROOT/'artifacts/v7_1/delivery_train_dev_001'
CACHE=OLD
BASE=ROOT/'artifacts/v7_1/smoke_002'
DEFAULT_OUT=ROOT/'artifacts/v7_1/expand256_train_dev_001'
CHAIN_OUT=ROOT/'artifacts/v7_1/expand256_train_chain_001'
BINDING=ROOT/'ff_prior_expand256_train/evidence/warm_binding.json'
CACHE_HASHES=read_json(BINDING)['cache_hashes']
METHODS=('structured','frozen_diffusion','learned_mean','expanded_diffusion')
PROTOCOL={
    'version':'expand256_single_warm_training_DEV_v1','TRAIN':'13966 + actual qualified expansion (128..256)','VAL':48,
    'warm_TRAIN':13966,'source_gate':'final expansion raw audit PASS, >=128 qualified, >=32 targets, >=26/32 execution replay, qualification/attempt >=0.5, no unresolved errors',
    'updates':6000,'batch':64,'training_seed':20263001,'lr':2e-4,'local_G_weight':.05,
    'warm_start':'exact delivered13966 model and codec; frozen normalization',
    'loss':'v MSE plus teacher-local pose surrogate; no new nonlinear world-gradient claim',
    'checkpoint_selection':'fixed final update; no VAL selection','sampling_distribution':'half uniform rows, half uniform origins',
    'G_reuse':'same original teacher ID/Z/task/codec, byte hashes and warm metadata matrix hash; no new anchors',
    'training_checkpoint_segment':20,'unknown_training_segment':'BLOCK without repeating unknown updates',
    'sampling_seeds':[20262119,20262123,20262129],'methods':list(METHODS),
    'original_DEV_tasks':4,'already_used_development_tasks':8,'candidates_per_task':5,
    'max_candidates':720,'max_selected_executions':144,'DDIM_steps':16,
    'diffusion_sampler':'same late4 for old13966 and new merged model','repair_queries':110,
    'candidate_wall_s':15.,'candidate_coupled_ceiling':113,'candidate_interval_ceiling':15872,
    'minimum_eligible_per_method':144,'minimum_execution_per_method':29,
    'selection':'min(score,candidate_id) among eligible; one execution, no replacement',
    'quality_or_efficiency_superiority_required':False,'wall_budget_s':14400.,'smoke_reserve_s':7200.,
    'cpu_threads':2,'max_output_bytes':20*1024**3,'new_expansion':False,'TEST':False,'pilot':False,
    'study':False,'strict_multimodal':False,'old_1024':False,'no_DEV_retuning_or_second_run':True}

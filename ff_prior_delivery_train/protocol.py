from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'artifacts/v7_1/delivery_dataset_128_001'
CACHE=ROOT/'artifacts/v7_1/world_endpoint_smoke_001'
OLD=ROOT/'artifacts/v7_1/qualified_origin_learn_002'
BASE=ROOT/'artifacts/v7_1/smoke_002'
DEFAULT_OUT=ROOT/'artifacts/v7_1/delivery_train_dev_001'
METHODS=('structured','frozen_diffusion','learned_mean','expanded_diffusion')
# Extracted and verified against paired_objectives_smoke_001/run.json's frozen
# source_files, not inferred from a newly computed cache digest alone.
CACHE_HASHES={
    'loaded_training_arrays.npz':'1e7204406a58c4196efedb367367e768b2caa5533604e7c8e655237a6f21e824',
    'training_input_audit.json':'8dc88016d7cbc2c8afb68086daa151687fd7a03ab2348d954e7b50f52ff0c343',
    'training_index.json':'59f5a55faaeef65b2c6607eca5dc4539cefee6c6b121db1a503c99aff1cce8c9'}
PROTOCOL={
    'version':'delivery_m3_single_training_and_DEV_1','TRAIN':13966,'VAL':48,
    'new_M2_rows':128,'additional_rows_since_warm_model':192,'warm_TRAIN':13774,
    'updates':6000,'batch':64,'training_seed':20262801,'lr':2e-4,'local_G_weight':.05,
    'warm_start':'exact verified 13774 model; frozen codec and condition normalization',
    'checkpoint_selection':'fixed final update; no VAL selection',
    'sampling_distribution':'half uniform rows, half uniform q17 origins',
    'G_reuse':'exact task + original anchor ID/Z + basis + checkpoint-bound matrix digest',
    'max_new_G_anchors':192,'new_G_only_for_new_rows':True,
    'training_checkpoint_segment':20,'unknown_training_segment':'BLOCK new training; never replay uncommitted updates',
    'sampling_seeds':[20262119,20262123,20262129],'methods':list(METHODS),
    'original_DEV_tasks':4,'already_used_development_tasks':8,'candidates_per_task':5,
    'max_candidates':720,'max_selected_executions':144,'DDIM_steps':16,
    'diffusion_sampler':'M1 selected late4 for both old and new weights',
    'other_methods':'original final-step hard repair for structured and weighted mean',
    'candidate_wall_s':15.,'candidate_coupled_ceiling':113,'candidate_interval_ceiling':15872,
    'repair_queries':110,'minimum_eligible_per_method':144,'minimum_execution_per_method':29,
    'selection':'min(score,candidate_id) among eligible; one execution; no replacement',
    'quality_or_efficiency_superiority_required':False,'raw_all_pass_required':False,
    'wall_budget_s':14400.,'smoke_reserve_s':7200.,'cpu_threads':2,'max_output_bytes':20*1024**3,
    'old_1024':False,'pilot':False,'study':False,'strict_multimodal':False,'TEST':False,
    'no_DEV_retuning_or_second_run':True,'no_new_world_gradient_ablation':True}

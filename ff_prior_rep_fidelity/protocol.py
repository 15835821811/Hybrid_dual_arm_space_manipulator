from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PACKAGE=ROOT/'ff_prior_rep_fidelity'
PREVIOUS=ROOT/'artifacts/v7_1/capture_trust_001'
SOURCE=ROOT/'artifacts/v7_1/sample_alignment_001'
BASE=ROOT/'artifacts/v7_1/smoke_002'
OUT=ROOT/'artifacts/v7_1/rep_fidelity_001'
P={'version':'fixed_temporal_enrichment_prepare_v1','task':'capture_5_obstacles',
   'split':'APPLICATION_DEV_DIAGNOSTIC','teacher':'07_bow_j11_1',
   'ranks':[6,10,14,19],'modes':['raw','legacy_prepare','preserve_prepare'],
   'basis':'frozen_TRAIN_rank6_then_terminal_unit_then_fixed_DCT_two_pass_Gram_Schmidt; rank19_identity',
   'legacy_input':'already_projected_raw_controls','legacy_smooth_weight':10.,
   'shrink_scales':[1.,.5,.25,.125,.0625,.03125,0.],
   'repair_iterations':0,'max_logical_candidates':12,'execution_limit':12,
   'execution_order':'rank_then_mode; exact_reference_duplicates_once; teacher_anchor_once_fresh',
   'validation_intervals':[256,512,1024,2048,4096,8192,16384],
   'coupled_cap':300,'integration_cap':3000000,'wall_s':7200,'cpu_threads':2,
   'output_bytes_cap':8*1024**3,'TRAIN_added':0,'training':False,'basis_fitted_to_DEV':False,
   'TEST':False,'physics_relaxed':False,'goal_relabel':False}

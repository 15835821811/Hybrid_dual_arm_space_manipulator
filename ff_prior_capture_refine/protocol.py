from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'artifacts/v7_1/sample_alignment_001'
PREVIOUS = ROOT/'artifacts/v7_1/capture_feasibility_001'
BASE = ROOT/'artifacts/v7_1/smoke_002'
OUT = ROOT/'artifacts/v7_1/capture_refine_001'
PACKAGE = ROOT/'ff_prior_capture_refine'
P = {'version':'feasible_dense_feedback_v1','task':'capture_5_obstacles',
 'split':'APPLICATION_DEV_DIAGNOSTIC','seeds':2,'teacher_iterations':12,'codec_repair_iterations':8,
 'seed_ids':['00_original_failed','07_bow_j11_1'],'seed_source':'previous_final_path_not_initial_seed',
 'sparse_base_grids':[512,1024,2048],'active_pair_cap':240,'active_buffer_m':.004,
 'dense_grids':[512,1024,2048,4096,8192,16384],
 'validation_intervals':[256,512,1024,2048,4096,8192,16384],
 'difference_rad':.0001,'trust_rad':.15,'solver_max_iterations':160,
 'line_scales':[1.,.5,.25,.125,.0625,.03125],
 'repair_clearance_m':.0002,'collision_scale_m':.003,
 'sparse_agreement_m':.001,'sparse_agreement_rad':.003490658503988659,
 'teacher_execution_limit':2,'execution_limit':5,
 'coupled_cap':12000,'integration_cap':24000000,'wall_s':14400,'cpu_threads':2,
 'output_bytes_cap':12*1024**3,'TRAIN_added':0,'training':False,'TEST':False,
 'physics_relaxed':False,'goal_relabel':False,'finite_failed_search_is_infeasibility_proof':False}

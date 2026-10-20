from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'artifacts/v7_1/qualified_expand256_001'
CONTROL=ROOT/'artifacts/v7_1/expand256_train_dev_001'
WARM=ROOT/'artifacts/v7_1/delivery_train_dev_001'
BASE=ROOT/'artifacts/v7_1/smoke_002'
DEFAULT_OUT=ROOT/'artifacts/v7_1/endpoint_jacobian_bounded_001'
PROTOCOL={'version':'group_jacobian_endpoint_bounded_1','TRAIN':14222,'VAL':48,'updates':6000,'batch':64,'lr':2e-4,
 'training_seed':20263001,'diagnostic_seed':20263201,'event_seed':20263202,'noise_times':[333,667,999],
 'diagnostic_targets':8,'diagnostic_cases':24,'event_targets':32,'world_events':60,'world_samples_per_event':2,
 'smooth_temperature':1.0,'smooth_loss':'tau*logmeanexp(squared_normalized_groups/tau)+.05*mean(squared_normalized_groups)',
 'gradient_method':'full102 squared-group Jacobian times analytic smooth weights at current central prediction',
 'gradient_checks':'unchanged direct scalar half-step and directional finite differences; no posthoc gate substitution',
 'joint_gradient':'equal event/control world gradients; diagnostic base also equal; ordinary M2 batch unchanged',
 'base_endpoint_counterfactual':'same evaluated fractions as combined; observational only; never changes acceptance',
 'world_every':100,'world_gradient_norm_ratio':.1,'line_scales':[1.,.5,.25],
 'endpoint_gradient_dimensions':102,'endpoint_difference_normalized':.01,'endpoint_convergence_intervals':512,
 'endpoint_gradient_half_tolerance':.02,'endpoint_gradient_direction_tolerance':.05,'queries_per_gradient':212,
 'M1_queries':12000,'M2_queries':27000,'total_queries':60000,'M1_wall_s':3600.,'M2_wall_s':9000.,
 'wall_budget_s':21600.,'max_output_bytes':20*1024**3,'val_seeds':[20263111,20263113],
 'sampling_seeds':[20262119,20262123,20262129],'max_new_candidates':180,'max_new_executions':36,
 'reuse_methods':{'structured':'structured','learned_mean':'learned_mean','control_A':'expanded_diffusion'},
 'diagnostic_minimum':16,'diagnostic_minimum_per_noise':4,'TEST':False,'expand':False,'pilot':False,
 'optimizer_state_rule':'accepted scaled parameter displacement; retain full combined AdamW moments and one step; otherwise restore and commit base proposal only',
 'acceptance_rule':'every event/control Q strictly decreases and previously passing groups stay <=1; finite raw joint-domain; no threshold relaxation'}

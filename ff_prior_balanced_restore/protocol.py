from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = Path(__file__).resolve().parent
OUT = ROOT/'artifacts/v7_1/balanced_restore_001'
OLD = ROOT/'artifacts/v7_1/horizon_restore_001'
BASE = ROOT/'artifacts/v7_1/smoke_002'
SOURCE = ROOT/'artifacts/v7_1/sample_alignment_001'
P = dict(version='balanced2R5C_full27s_323_restoration_v1',
    task_id='balanced_2R5C',
    duration_s=27.,hold_s=.5,candidate_denominator=1,execution_units=1,
    iterations=28,fit_samples=545,fit_solver_iterations=400,solver_iterations=180,
    dense_grids=[512,1024,2048,4096,8192,16384],
    sparse_grids=[512,1024,2048],active_cap=240,active_buffer_m=.004,
    geometry_step_m=.002,margin_m=.0002,distance_scale_m=.003,
    sparse_agreement_m=.001,sparse_agreement_rad=.003490658503988659,
    difference_rad=.0001,initial_radius_rad=.0375,max_radius_rad=.15,
    min_radius_rad=.001171875,line_scales=[1.,.5,.25,.125,.0625,.03125],
    ratio_accept=.1,goal_weight=.05,fit_smooth_weight=.0001,motion_fraction=.995,
    iteration_query_reserve=366,iteration_interval_reserve=946016,
    coupled_cap=11000,unit_coupled_cap=11000,restore_coupled_cap=10600,
    integration_cap=29000000,unit_integration_cap=29000000,restore_interval_cap=28500000,
    wall_s=15000,unit_wall_s=15000,restore_wall_s=13200,execution_reserve_s=600,
    output_bytes_cap=12*1024**3,native_steps_cap=27500,cpu_threads=2,
    diagnostics_wall_s=600,visualization_wall_s=600,render_frames=420,render_width=960,render_height=720,render_fps=15,
    prior_D3_coupled=46454,prior_D3_worker_s=2458.484,prior_D3_bytes=1224791218,
    D3_coupled_cap=60000,D3_wall_s=21540,D3_output_bytes=20*1024**3,D3_native_cap=165000,
    future_dynamic_native_reserve=27000,future_dynamic_coupled_reserve=2000,
    future_dynamic_wall_reserve_s=1200,future_dynamic_output_bytes=2*1024**3,evidence_bytes_reserve=1024**3,
    new_neural_samples=0,training_updates=0,TRAIN_added=0,TEST=False,
    advantage_comparison=False,physics_relaxed=False,goal_relabel=False,
    qualification='APP_DEV single engineering unit; historical Diffusion-derived full323 reference reused as new-scene unqualified guess',
    no_automatic_retry=True)

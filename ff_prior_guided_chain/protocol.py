from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts/v7_1/guided_diffusion_chain_001'
P = dict(version='frozen_diffusion_full_horizon_inherited_momentum_v1',
    candidate_chains=1, candidates_per_phase=1, retries=0,
    coupled_cap=40000, interval_cap=90000000, native_cap=83500,
    neural_cap=48, bytes_cap=8*1024**3, wall_s=43200,
    geometry_cap=3000000000, mass_kinematic_cap=400000000,
    implementation_s=7200, planner_s=21600, native_s=3600,
    ddim_steps=16, guidance_steps=[12,13,14,15], solver_iterations=180,
    fd_h=1e-4, probe_relative_tolerance=.02,
    dense_grids=[512,1024,2048,4096,8192,16384], sparse_grids=[512,1024,2048],
    geometry_step=.002, position_tolerance=.005, rotation_tolerance=0.017453292519943295,
    active_distance=.008, geometry_rows=512, torque_rows=256,
    line_scales=[1.,.5,.25,.125], trust_radius=.0375,
    projection_smooth_weight=1e-4, motion_fraction=.995,
    guidance_regularization=1e-4, cpu_threads=2,
    phase_seeds=dict(A=2026102601,B=2026102602,C=2026102603),
    phase_intervals=dict(A=13750,B=13500,C=14500),
    phase_durations=dict(A=27.5,B=27.,C=29.),
    initial='ff_prior_balanced_restore/evidence/initial_state.npz',
    initial_sha256='5ad6c16207374d7368f2cff96d7e4ad7bd6c4aa92cc35e20aa2db451c28e91e6',
    static_model='artifacts/v7_1/sample_alignment_001/learning/aligned',
    temporal_model='artifacts/v7_1/temporal_learning_001/learning/temporal_diffusion',
    temporal_codec='artifacts/v7_1/temporal_learning_001/codec.json',
    frozen_weights=True, training=False, expansion=False, TEST=False,
    advantage_comparison=False, physics_gate_changes=[], actual_resets=0,
    actual_initializations=1,replay_initializations=1,subtract_initial_robot_momentum=False)

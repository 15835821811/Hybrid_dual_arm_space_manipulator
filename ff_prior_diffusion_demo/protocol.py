"""One public-input Diffusion engineering episode; no advantage benchmark."""
from pathlib import Path
from ff_prior_static_fullchain_v2.protocol import P as ORIGINAL
ROOT=Path(__file__).resolve().parents[1]
PACKAGE=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/v7_1/diffusion_demo_001'
SOURCE=ROOT/'artifacts/v7_1/sample_alignment_001'
BASE=ROOT/'artifacts/v7_1/smoke_002'
CODEC=ROOT/'artifacts/v7_1/control_transfer_bounded_001/codec.json'
P=dict(ORIGINAL,version='diffusion_single_episode_full_visuals_v1',
    tasks=['capture_3_obstacles'],methods=['diffusion_geometry'],seeds=[20264611],
    candidate_denominator=5,execution_units=1,coupled_cap=12000,
    integration_cap=29000000,native_steps_cap=27500,wall_s=7200,
    unit_wall_s=1800,render_frames=400,render_width=960,render_height=720,
    render_fps=15,visualization_wall_s=1200,
    output_bytes_cap=8*1024**3,full_integration_replay_required=True,
    primary_objective='one full public-input diffusion system episode and visual evidence',
    comparison_runs=0,new_training=0,full_static_dynamic_chain_claim=False)

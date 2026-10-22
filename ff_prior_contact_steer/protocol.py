from pathlib import Path
from ff_prior_world_steer.protocol import P as PREVIOUS

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = Path(__file__).resolve().parent
OUT = ROOT/'artifacts/v7_1/contact_steer_001'
OLD = ROOT/'artifacts/v7_1/world_steer_001'
SOURCE = ROOT/'artifacts/v7_1/sample_alignment_001'
BASE = ROOT/'artifacts/v7_1/smoke_002'
P = dict(PREVIOUS, version='whole_edge_contact_jacobian_exact_context_dedup_v1',
    candidate_denominator=3, execution_units=3, coupled_cap=20860,
    unit_coupled_cap=6920, integration_cap=35000000, unit_integration_cap=11000000,
    wall_s=18000, unit_wall_s=5600, search_wall_s=3600, repair_wall_s=1500,
    output_bytes_cap=18*1024**3, native_steps_cap=82500,
    search_coupled_cap=4000, search_integration_cap=2200000,
    repair_coupled_cap=2800, repair_integration_cap=8000000, repair_iterations=6,
    validation_coupled_cap=40, validation_integration_cap=200000,
    assembly_coupled_cap=40, assembly_integration_cap=400000,
    search_attempt_cap=50, accepted_node_cap=160,
    contact_grid=256, contact_phase_stride=8, contact_active_cap=96,
    contact_buffer_m=.012, contact_distance_scale_m=.003, contact_margin_m=.0002,
    contact_difference_rad=.0001, contact_trust_rad=.12,
    contact_solver_iterations=120, contact_duration_s=1.,
    contact_world_growth=1.30, contact_tangent_rad=.08,
    contact_parent_time_weight=.2, contact_query_reserve=110,
    contact_interval_reserve=45000, contact_line_scales=[1.,.5,.25],
    contact_variant_ids=[0,1,-1], visualization_wall_s=600,
    inherited_cycle_coupled=37414, inherited_cycle_wall_s=1733.609,
    inherited_cycle_bytes=585993194, remaining_cycle_coupled=22586,
    dynamic_future_native_reserve=81000,
    new_network_samples=0, historical_neural_proposals_reused=3,
    reuse_level='HISTORICAL_MODEL_SCREEN_PREFIX_NOT_EXECUTED',
    no_mechanism_retry=True)

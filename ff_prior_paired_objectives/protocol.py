from pathlib import Path
from ff_prior_world_endpoint.protocol import PROTOCOL as PRIOR
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'artifacts/v7_1/world_endpoint_smoke_001'
WARM=ROOT/'artifacts/v7_1/qualified_origin_learn_002'
DEFAULT_OUT=ROOT/'artifacts/v7_1/paired_objectives_smoke_001'
METHODS=['structured','frozen_diffusion','learned_mean','expanded_diffusion']
ARMS={'A_v':[0.,0.],'B_local':[.05,0.],'C_world':[0.,.1],'D_both':[.05,.1]}
PROTOCOL=dict(PRIOR,version='paired_objectives_2x2_v1',training_seed=20262501,
    world_noise_seed=20262502,old_TRAIN=13838,new_TRAIN=0,arms=ARMS,
    single_step_cases=8,single_step_targets=4,parameter_difference=.001,
    parameter_direction_tolerance=.05,parameter_descent_steps=[.001,.0003,.0001],
    optimizer='fresh AdamW per arm; lr2e-4, default weight_decay.01; clip1',
    same_random_sequence='CoverageSampler seed20262503; per-step torch generator seed20262501+step',
    local_loss='frozen task-anchor G normalized by fixed codec latent_scale; every batch',
    TRAIN_gate='numeric8/8, four complete paired arms; C/D median endpoint ratio vs warm<=.8 and vs B<=1',
    TRAIN_selection='eligible C/D minimum median after/warm ratio; tie lexicographic; before DEV',
    conditional_TEST_gate='BLOCKED_BY_CURRENT_USER_SCOPE; no TEST regardless of results',
    single_step_scope='8 fixed TRAIN probes; optimizer-step outcomes diagnostic, not forced descent',
    full_goal_auto_complete=False)

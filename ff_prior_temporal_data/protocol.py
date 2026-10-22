from pathlib import Path
from ff_prior_dynamic_margin.protocol import P as REFERENCE
ROOT=Path(__file__).resolve().parents[1]
PACKAGE=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/v7_1/temporal_data_001'
SOURCE=ROOT/'artifacts/v7_1/waypoint_moving_gate_001/freeze.json'
OLD_REFERENCE=ROOT/'artifacts/v7_1/dynamic_reference_001'
OLD_MARGIN=ROOT/'artifacts/v7_1/dynamic_margin_001'
P=dict(REFERENCE,stage='P3a_existing_dynamic_data_qualification',threads=1,
       inventory_train=32,inventory_validation=8,selected_train=8,selected_validation=4,
       executions=8,reuse_train_indices=[0,1,2,3],wall_s=3000,max_bytes=5*1024**3,
       max_native_replay_steps=216000,training=False,expansion=False,TEST=False,
       P3_total_wall_s=21600,P3_total_bytes=20*1024**3,
       gate_train=8,gate_validation=4,replace_failed_selection=False)
for unused in ['new_execution_case','reuse_cases','P2b_remaining_new_steps']:
    P.pop(unused,None)

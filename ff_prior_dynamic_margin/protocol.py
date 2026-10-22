from pathlib import Path
from ff_prior_dynamic_reference.protocol import P as ORIGINAL
ROOT=Path(__file__).resolve().parents[1]
PACKAGE=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/v7_1/dynamic_margin_001'
SOURCE=ROOT/'artifacts/v7_1/waypoint_moving_gate_001/freeze.json'
PREVIOUS=ROOT/'artifacts/v7_1/dynamic_reference_001'
P=dict(ORIGINAL,stage='P2a_single_response_margin_repair',
       acceleration_fraction=0.9695632054405793,executions=1,
       max_native_replay_steps=27000,wall_s=600,max_bytes=1024**3,
       new_execution_case=0,reuse_cases=[1,2,3],P2b_remaining_new_steps=297000)

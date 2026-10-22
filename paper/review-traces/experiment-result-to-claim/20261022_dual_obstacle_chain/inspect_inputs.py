from pathlib import Path
import json, zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[4]
RUN=ROOT/'artifacts/v7_1/dual_obstacle_chain_001'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
r=read(RUN/'run.json')
print('RUN KEYS',r.keys(),'SOURCE',len(r['identity']['source_hashes']),'INPUT',len(r['identity']['binding']['files']),'HEAD',r['producer_head'])
print('BINDINGS',json.dumps(r['identity']['binding'],indent=2))
for name in ['static/budget.json','dynamic/budget.json','pending.json','initial_scene_audit.json','dynamic/contract.json','static/units/u00/candidates/04/provenance.json','static/units/u00/candidates/04/validated/validation.json']:
 print(name,json.dumps(read(RUN/name),indent=2)[:15000])
for name in ['static/units/u00/execution/native_trace.npz','static/units/u00/execution/replay_trace.npz','dynamic/initial.npz','dynamic/execution/native_trace.npz','dynamic/execution/replay_trace.npz','dynamic/execution/geometry.npz']:
 with np.load(RUN/name) as a:print(name,{k:(a[k].shape,str(a[k].dtype)) for k in a.files})

from pathlib import Path
import json, numpy as np
root=Path(__file__).resolve().parents[4]
run=root/'artifacts/v7_1/horizon_restore_001'
def compact(v):
    if isinstance(v,dict):return {k:compact(x) for k,x in v.items() if k not in ('source_hashes','files','Z','solved_u','returned_u','returned_gradient','clearance_path_m')}
    if isinstance(v,list) and len(v)>35:return {'length':len(v),'first':compact(v[0]),'last':compact(v[-1])}
    if isinstance(v,list):return [compact(x) for x in v]
    return v
for name in ['run.json','physical_contract.json','ledger.json','units/u00/execution/report.json','units/u00/restore/iter/00/record.json','units/u00/restore/iter/00/probes.json','units/u00/execution/initial_state.json','scenes/capture_5_obstacles/initial_audit.json']:
    print('\nFILE',name,json.dumps(compact(json.loads((run/name).read_text())),ensure_ascii=False))
for name in ['units/u00/fit.npz','units/u00/restore/initial.npz','units/u00/restore/iter/00/jacobian.npz','units/u00/validated/grid_4096.npz','units/u00/execution/native_trace.npz','units/u00/execution/replay_trace.npz','units/u00/execution/initial_state.npz']:
    with np.load(run/name) as a:print('ARRAYS',name,{k:(a[k].shape,str(a[k].dtype)) for k in a.files})

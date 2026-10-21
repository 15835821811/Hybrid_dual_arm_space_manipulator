"""Read-only provenance of the regressed TRAIN case; no physical queries."""
import json
import numpy as np
from ff_prior.storage import read_json,file_digest,digest
from ff_prior.spline import Reference
from ff_prior.contracts import MODEL_GATES
from ff_prior_control_transfer.protocol import ROOT,OUT,DATA
from pathlib import Path

dest=ROOT/'paper/review-traces/experiment-result-to-claim/20261021_control_transfer_run01'
index=read_json(DATA/'training_input_audit.json')['records']
j=next(i for i,e in enumerate(index) if e['sample_id']=='coverage_train_00288');entry=index[j]
p=ROOT/entry['record'];r=read_json(p);directory=p.parent.parent
array=directory/r['arrays'];initial=directory/r['complete_initial_state'];genpath=directory/r['generation_proof'];gen=read_json(genpath)
assert file_digest(p)==entry['record_sha256']
assert file_digest(array)==entry['arrays_sha256']==r['arrays_sha256']
assert file_digest(initial)==r['initial_state_sha256']==gen['complete_initial_state_sha256']
assert r['condition']==r['task']['condition'] and r['training_eligible'] and r['split']=='TRAIN'
assert r['validation_level']=='MODEL_VALID' and r['execution_status']=='NOT_RUN'
assert all(r['validation']['gates'][g] for g in MODEL_GATES)
assert max(r['validation']['frozen_goal_position_error_m'])<=.005
assert max(r['validation']['frozen_goal_rotation_error_rad'])<=np.deg2rad(1)
assert gen['admitted'] and not r['goal_relabelled'] and not gen['goal_relabelled']
with np.load(array) as a,np.load(DATA/'loaded_training_arrays.npz') as cache:
    teacher=a['Z'].copy();assert np.array_equal(teacher,cache['Z'][j])
    assert Reference(teacher,r['condition']['initial_features'],r['condition']['duration']).reference_hash==r['reference_hash']
fixed=read_json(OUT/'frozen_tasks.json')['diagnostic'];assert fixed[6]==r
samples=[]
for phase in ('before','after'):
    d=OUT/'TRAIN_diagnostic'/phase/'12';identity=read_json(d/'identity.json');v=read_json(d/'validation.json')
    assert identity['task_hash']==r['task']['task_hash']==v['task_hash'] and identity['seed']==20270213
    with np.load(d/'controls.npz') as a:
        assert np.array_equal(a['teacher'],teacher) and np.array_equal(a['raw'],a['prepared'])
        rmse=float(np.sqrt(np.mean((a['raw']-teacher)**2)))
    samples.append(dict(phase=phase,control_RMSE_rad=rmse,eligible=v['eligible'],position_mm=max(v['validation']['frozen_goal_position_error_m'])*1000,rotation_deg=float(np.rad2deg(max(v['validation']['frozen_goal_rotation_error_rad'])))))
result={'status':'PASS','sample_id':r['sample_id'],'task_hash':r['task']['task_hash'],'seed':20270213,'original_record':p.relative_to(ROOT).as_posix(),'cached_teacher_exactly_equals_bound_original_Z':True,'task_condition_target_and_initial_state_identity_match':True,'original_teacher_level':r['validation_level'],'original_teacher_execution_status':r['execution_status'],'original_goal_source':r['goal_source'],'original_teacher_position_mm':max(r['validation']['frozen_goal_position_error_m'])*1000,'original_teacher_rotation_deg':float(np.rad2deg(max(r['validation']['frozen_goal_rotation_error_rad']))),'paired_diagnostic':samples,'interpretation':'No teacher/target cache mismatch found. Lower physical-control RMSE coincides with worse world endpoint on this fixed sample; no cause isolation and no new teacher propagation was performed.'}
(dest/'regression_provenance.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
inputs=read_json(dest/'inputs.json')
for path in (p,array,initial,genpath,Path(__file__).resolve(),ROOT/'ff_prior_control_transfer/docs/ANALYSIS.md'):
    inputs[path.relative_to(ROOT).as_posix()]={'sha256':file_digest(path),'bytes':path.stat().st_size,'mtime_ns':path.stat().st_mtime_ns}
(dest/'inputs.json').write_text(json.dumps(inputs,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(result,ensure_ascii=False,indent=2))

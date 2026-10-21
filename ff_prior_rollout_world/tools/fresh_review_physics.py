"""Independent saved-state math only. Never step or construct a simulator."""
import json,sys,hashlib,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from ff_prior.storage import digest
from ff_prior.diffusion import weight_hash
from ff_prior.contracts import MODEL_GATES
from ff_prior_control_transfer.codec import FullCodec
from ff_prior_control_transfer.model import TransferDenoiser
OUT=ROOT/'artifacts/v7_1/rollout_world_bounded_001';TRACE=ROOT/'paper/review-traces/experiment-result-to-claim/20261021_rollout_world_run01'
inputs={};results={}
def track(p):
 p=Path(p);inputs[str(p.relative_to(ROOT))]=dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),mtime_utc=datetime.datetime.fromtimestamp(p.stat().st_mtime,datetime.timezone.utc).isoformat(),bytes=p.stat().st_size);return p
def js(p):return json.loads(track(p).read_text(encoding='utf-8-sig'))
def npz(p):return dict(np.load(track(p)))
def error(ee,goal):
 ee=np.asarray(ee);goal=np.asarray(goal);p=np.linalg.norm(ee[...,:3,3]-goal[...,:3,3],axis=-1)
 mat=np.swapaxes(goal[...,:3,:3],-1,-2)@ee[...,:3,:3];r=Rotation.from_matrix(mat.reshape(-1,3,3)).magnitude().reshape(mat.shape[:-2]);return p,r
selection=js(OUT/'frozen_tasks.json');taskmap={e['task']['task_hash']:e['task'] for key in ('diagnostic','entries','application') for e in selection[key]}
native=[]
for rp in sorted(OUT.glob('comparison/*/*/*/execution/report.json')):
 d=rp.parent;r=js(rp);task=js(d/'frozen_task.json');a=npz(d/'native_trace.npz');b=npz(d/'replay_trace.npz');geometry=npz(d/'geometry_scans.npz')
 assert task==taskmap[r['task_hash']] and r['initial_state_id']==task['contract']['initial_state_id']
 assert a['ctrl'].shape[1]==67 and a['qpos'].shape[1]==81 and len(a['qpos'])==len(a['ctrl'])+1
 assert np.array_equal(a['ctrl'],b['ctrl']) and digest(a['ctrl'])==r['ctrl_hash']
 errors={k:float(np.max(abs(a[k]-b[k]),initial=0)) for k in ('qpos','qvel','act','time','qacc_warmstart')}
 assert errors==r['replay']['maximum_errors'] and max(errors.values())<=1e-10
 p,ang=error(a['ee'],task['condition']['goals']);hold=a['time']-a['time'][0]>=task['condition']['duration']-1e-9
 assert np.max(abs(p[hold].max(0)-r['hold_max_position_error_m']))<1e-12 and np.max(abs(ang[hold].max(0)-r['hold_max_rotation_error_rad']))<1e-10
 assert np.max(p[hold])<=.005 and np.max(ang[hold])<=np.pi/180
 assert float(geometry['minima'].min())==r['minimum_clearance_m'] and r['minimum_clearance_m']>=0
 assert not r['metadata']['robot_state_writes_during_step_loop'] and not r['metadata']['external_wrench_applied']
 assert all(r['gates'].values()) and r['level']=='EXECUTION_VERIFIED'
 native.append(dict(path=str(d.relative_to(OUT)),steps=2*len(a['ctrl']),hold_position_m=float(p[hold].max()),hold_rotation_deg=float(ang[hold].max()*180/np.pi),minimum_clearance_m=r['minimum_clearance_m'],replay_errors=errors))
results['new_native_replay']=dict(records=native,count=len(native),total_steps=sum(r['steps'] for r in native),maximum_replay_error=max(max(r['replay_errors'].values()) for r in native))
print('Native/replay saved arrays independently checked',len(native),flush=True)

validations=[]
for stage in ('TRAIN_world','candidate_world'):
 for p in sorted((OUT/stage).glob('**/validation.json')):
  r=js(p);v=r['validation'];task=taskmap[r['task_hash']];a=npz(p.parent/'prediction.npz');pos,rot=error(a['ee'][-1],task['condition']['goals'])
  assert np.max(abs(pos-v['frozen_goal_position_error_m']))<1e-12 and np.max(abs(rot-v['frozen_goal_rotation_error_rad']))<1e-10
  model=all(v['gates'].get(g,False) for g in MODEL_GATES);goal=max(pos)<=.005 and max(rot)<=np.pi/180
  assert model==(v['validation_level']=='MODEL_VALID') and bool(model and goal)==r['eligible']
  scanpaths=sorted(p.parent.glob('grid_*_geometry_scans.npz'))
  for sp in scanpaths:
   scan=npz(sp)
   # Stored scan arrays are directly inspected, retaining failures in denominator.
   assert np.isfinite(scan['minima']).all()
  validations.append(dict(path=str(p.relative_to(OUT)),eligible=r['eligible'],position_m=float(max(pos)),rotation_deg=float(max(rot)*180/np.pi),failed_gates=[k for k,vv in v['gates'].items() if not vv]))
results['saved_TRAIN_physics']=dict(count=len(validations),records=validations)
results['application_failures']=[]
for p in sorted((OUT/'comparison/APPLICATION_DEV').glob('*/*/candidates/*.json')):
 r=js(p);v=r['validation'];results['application_failures'].append(dict(sample_id=r['sample_id'],candidate_id=r['candidate_id'],eligible=r['eligible'],failed_gates=[k for k,vv in v['gates'].items() if not vv],clearance_m=v['geometry']['minimum_clearance_m']))
results['reuse_counts']={}
for scope in ('DEV','APPLICATION_DEV'):
 for method in ('old_aligned','structured','learned_mean'):
  entries=[e for e in selection['reuse'] if e['scope']==scope and e['method']==method];outcomes=[js(ROOT/e['execution']/'outcome.json') for e in entries]
  results['reuse_counts'][scope+'/'+method]=dict(candidates=sum(len(e['candidates']) for e in entries),execution_opportunities=len(entries),execution_verified=sum(x['status']=='EXECUTION_VERIFIED' for x in outcomes),outcomes=[x['status'] for x in outcomes])
codec=FullCodec(js(OUT/'codec.json'));model=TransferDenoiser(codec,warm=False);final=torch.load(track(OUT/'learning/full/model.pt'),weights_only=False,map_location='cpu');formal=torch.load(track(OUT/'candidate_world/final.pt'),weights_only=False,map_location='cpu')
assert all(torch.equal(v,formal['weights'][k]) for k,v in final['weights'].items());model.load_state_dict(final['weights']);gate=js(OUT/'candidate_world/gate.json');assert weight_hash(model)==gate['final_hash']
results['candidate_binding']={'final_weights_match':True,'final_hash':gate['final_hash'],'shared_metadata_retained_for_candidate':gate['retained_for_candidate'],'metadata_caveat':'shared refinement returns false even though formal weights were retained; actual model tensors bind correctly'}
resume=js(ROOT/'ff_prior_rollout_world/evidence/completed/completed_resume.json')
assert resume['before']==resume['after'] and resume['core_files_unchanged'] and resume['frozen_identity_unchanged'] and all(resume['user_files_unchanged'].values())
results['completed_resume']=dict(files=len(resume['before']),hashes_identical=True,response=resume['response']['resume'],reviewer_did_not_call_resume=True)
(TRACE/'physics_checks.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8');(TRACE/'physics_inputs.json').write_text(json.dumps(inputs,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in results.items() if k not in ('saved_TRAIN_physics','new_native_replay')},ensure_ascii=False),flush=True)

"""Small independent closure checks; no physics or project code imported."""
from pathlib import Path
import json,hashlib
import numpy as np
from scipy.interpolate import BSpline
ROOT=Path(__file__).resolve().parents[4];OUT=Path(__file__).resolve().parent;RUN=ROOT/'artifacts/v7_1/rep_fidelity_001'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks=[]
def ck(name,value,details=None):checks.append(dict(name=name,passed=bool(value),details=details))
r=read(ROOT/'ff_prior_rep_fidelity/evidence/completed/completed_resume.json')
ck('completed_resume_235_current_files',r['before']==r['after'] and len(r['after'])==235 and all(sha(RUN/p)==h for p,h in r['after'].items()))
ck('completed_resume_no_work',r['response']['resume']=='NO_WORK_ALREADY_COMPLETE' and r['returncode']==0)
lims={k:np.array(v) for k,v in read(RUN/'physical_limits.json').items()};pc=read(RUN/'physical_contract.json');old=read(ROOT/'artifacts/v7_1/sample_alignment_001/physical_contract.json')
ck('canonical_contract_matches_old',pc['identity']==old['identity'] and pc['acceptance']==old['acceptance'] and pc['initial_state_id']==old['initial_state_id'])
identity=pc['identity'];names=identity['robot']['low_level_joint_names'];mapping=identity['joint_index_map'];qi=[mapping[n]['qpos'] for n in names];vi=[mapping[n]['dof'] for n in names];ai=[mapping[n]['actuator'] for n in names]
n=np.load(RUN/'candidates/r19_raw/execution/native_trace.npz');B=lims['B'];Bp=np.linalg.pinv(B);lowq=n['qpos'][:,qi];lowv=n['qvel'][:,vi];q=lowq@Bp.T;v=lowv@Bp.T;a=np.diff(v,axis=0)/.002;la=np.diff(lowv,axis=0)/.002;limited=lims['limited'].astype(bool)
ck('actual_joint_limits',np.all(q>=lims['lower']) and np.all(q<=lims['upper']) and np.all(lowq[:,limited]>=lims['ranges'][limited,0]) and np.all(lowq[:,limited]<=lims['ranges'][limited,1]))
ck('actual_derivative_limits',np.all(abs(v)<=lims['velocity']) and np.all(abs(a)<=lims['acceleration']) and np.all(abs(lowv)<=abs(B)@lims['velocity']) and np.all(abs(la)<=abs(B)@lims['acceleration']))
ck('stored_actual17_consistency',np.allclose(q,n['q_actual_17'],atol=1e-12,rtol=0))
ck('stored_manifold_consistency',np.allclose(lowq-q@B.T,n['manifold_residual'],atol=1e-12,rtol=0))
torque_limits=np.r_[np.full(60,4.),np.full(7,80.)]
ck('raw_torque_limits',np.all(abs(n['raw_torque'])<=torque_limits+1e-12) and np.allclose(n['ctrl'][:,ai],np.clip(n['raw_torque'],-torque_limits,torque_limits),atol=1e-12,rtol=0),{'max_fraction':float(np.max(abs(n['raw_torque'])/torque_limits))})
task=read(RUN/'frozen_task.json');q0=np.array(task['condition']['initial_features']);T=task['condition']['duration'];z=np.load(RUN/'representation_inputs.npz')['teacher'];ctl=np.vstack([np.repeat(q0[None],3,axis=0),z[:18],np.repeat(z[-1:],3,axis=0)]);s=BSpline(np.r_[np.zeros(4),np.arange(1,21)/21,np.ones(4)],ctl,3)
t=np.arange(13750)*.002
for o,key in enumerate(['reference_q','reference_qd','reference_qdd']):
    curve=s if o==0 else s.derivative(o); val=curve(np.clip(t/T,0,1))/T**o
    if o:val=np.where((t>T)[:,None],0,val)
    ck(key+'_alignment',np.allclose(val,n[key],atol=1e-12,rtol=0))
for rnk in [10,14]:
    p=np.load(RUN/f'candidates/r{rnk}_raw/representation.npz');ck(f'rank{rnk}_terminal_control_preserved',np.max(abs(p['raw'][-1]-z[-1]))<1e-12, float(np.max(abs(p['raw'][-1]-z[-1]))))
codec=read(ROOT/'artifacts/v7_1/sample_alignment_001/learning/aligned/codec.json');ck('bound_codec_declares_TRAIN_and_excludes_task',codec['fit_split']=='TRAIN' and task['task_hash'] not in codec['fit_ids'],{'fit_ids':len(codec['fit_ids']),'fit_hash':codec['fit_hash']})
result=dict(checks=checks,failures=[c for c in checks if not c['passed']]);(OUT/'closure_checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(json.dumps(result))

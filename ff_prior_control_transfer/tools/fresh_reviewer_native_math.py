"""Additional native saved-array arithmetic, no robot creation or simulation."""
from pathlib import Path
import json
import numpy as np
from ff_prior.storage import read_json,digest,file_digest
from ff_prior.spline import Reference
from model_test.angle_convention import continuum_mujoco_map_60_to_10
from ff_prior_control_transfer.protocol import ROOT,OUT,SOURCE

dest=ROOT/'paper/review-traces/experiment-result-to-claim/20261021_control_transfer_run01'
contract=read_json(OUT/'physical_contract.json');identity=contract['identity'];profile=contract['acceptance']
mapping=sorted(identity['joint_index_map'].values(),key=lambda r:r['actuator'])
qids=[v['qpos'] for v in mapping];vids=[v['dof'] for v in mapping]
B=np.zeros((67,17));B[:60,:10]=continuum_mujoco_map_60_to_10();B[60:,10:]=np.eye(7);inverse=np.linalg.pinv(B)
vel=np.r_[np.full(10,.8),np.full(7,1.2)];acc=np.r_[np.full(10,2.5),np.full(7,4.)]
torque=np.r_[np.full(60,4.),np.full(7,80.)]
results=[]
for p in sorted(OUT.rglob('execution/report.json')):
    d=p.parent;r=read_json(p);task=read_json(d/'frozen_task.json');init=read_json(d/'initial_state.json')
    with np.load(d/'initial_state.npz') as a:
        assert digest({'values':a['values'],'model_hash':init['model_hash'],'state_mask':init['state_mask']})==init['state_id']==task['contract']['initial_state_id']
    if 'reconstruction' in d.parts:Z=read_json(d.parent/'validation.json')['Z']
    else:Z=read_json(d.parent/'candidates'/f"{read_json(d/'selection.json')['selected_id']:02d}.json")['Z']
    ref=Reference(Z,task['condition']['initial_features'],task['condition']['duration'])
    with np.load(d/'native_trace.npz') as a:
        assert all(np.isfinite(a[k]).all() for k in a.files)
        time=a['time']-a['time'][0];dt=.002
        assert np.allclose(np.diff(time),dt,rtol=0,atol=1e-12)
        assert abs(time[-1]-task['condition']['duration']-.5)<1e-9 and time[-1]<=30
        assert len(a['ctrl'])==r['metadata']['physics_steps']==r['metadata']['planned_steps']
        q=a['qpos'][:,qids]@inverse.T;v=a['qvel'][:,vids]@inverse.T
        assert np.max(abs(q-a['q_actual_17']))<1e-12
        track=float(abs(q-ref.evaluate(time)).max());manifold=float(abs(a['qpos'][:,qids]-q@B.T).max())
        assert abs(track-r['tracking_max_rad'])<1e-12 and track<=profile['tracking_error_rad']
        assert abs(manifold-r['manifold_max_rad'])<1e-12 and manifold<=profile['manifold_residual_rad']
        assert np.all(q>=identity['planner_lower']) and np.all(q<=identity['planner_upper'])
        assert np.all(np.abs(v)<=vel) and np.all(np.abs(np.diff(v,axis=0)/dt)<=acc)
        assert np.all(abs(a['qvel'][:,vids])<=abs(B)@vel)
        assert np.all(abs(np.diff(a['qvel'][:,vids],axis=0)/dt)<=abs(B)@acc)
        for order,key in enumerate(('reference_q','reference_qd','reference_qdd')):
            assert np.max(abs(a[key]-ref.evaluate(np.arange(len(a['ctrl']))*dt,order)))<1e-12
        assert np.all(abs(ref.evaluate(time,1))<=vel) and np.all(abs(ref.evaluate(time,2))<=acc)
        assert np.all(abs(a['raw_torque'])<=torque+1e-12) and np.max(abs(a['ctrl']-np.clip(a['raw_torque'],-torque,torque)))<=1e-12
        momentum=a['base_generalized_momentum'];peaks=[np.max(np.linalg.norm(momentum[:,s],axis=1)) for s in (slice(0,3),slice(3,6))]
        assert np.max(abs(np.asarray(peaks)-r['momentum_peak_linear_angular']))<1e-12
        threshold=profile['momentum_atol']+profile['momentum_relative_tolerance']*np.asarray(r['momentum_exchange_scale_linear_angular'])
        assert np.array_equal(threshold,r['momentum_threshold_linear_angular']) and np.all(peaks<=threshold)
        assert abs(float(abs(momentum).max())-r['zero_momentum_max_residual'])<1e-12
        max_base=float(np.linalg.norm(a['base'][:,:3]-a['base'][0,:3],axis=1).max())
        assert max_base==r['maximum_base_translation_m']
    with np.load(d/'geometry_scans.npz') as a:
        scans=read_json(d/'geometry_scans.json')
        assert np.array_equal(a['minima'],[s['minimum_clearance_m'] for s in scans])
        assert len(scans)==2*r['metadata']['physics_steps']+1
        assert set(s['fraction'] for s in scans)=={0.,.5}
    results.append({'path':d.relative_to(OUT).as_posix(),'tracking_max_rad':track,'manifold_max_rad':manifold,'momentum_peak_linear_angular':list(map(float,peaks)),'complete_initial_state_digest_pass':True,'finite_time_reference_derivatives_torque_momentum_recomputed':True})

assert len(results)==3
result={'status':'PASS','native_arithmetic':results,'limitations':'Saved native arrays only. No fresh collision, mass matrix, kinematic or dynamics queries. Geometry distances and momentum exchange scales are source-traced and cross-checked against saved scans/reports.'}
(dest/'native_math.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
inputs=read_json(dest/'inputs.json')
for p in (Path(__file__).resolve(),ROOT/'model_test/angle_convention.py'):
    inputs[p.relative_to(ROOT).as_posix()]={'sha256':file_digest(p),'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns}
(dest/'inputs.json').write_text(json.dumps(inputs,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(result,ensure_ascii=False,indent=2))

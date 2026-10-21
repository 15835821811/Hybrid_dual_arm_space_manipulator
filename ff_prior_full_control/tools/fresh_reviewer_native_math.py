"""Additional native saved-array arithmetic, no robot creation or simulation."""
from pathlib import Path
import json
import numpy as np
from ff_prior.storage import read_json,digest,file_digest
from ff_prior.spline import Reference
from model_test.angle_convention import continuum_mujoco_map_60_to_10
from ff_prior_full_control.protocol import ROOT,OUT,SOURCE

dest=ROOT/'paper/review-traces/experiment-result-to-claim/20261021_full_control_run01'
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

def dist(records):
    pos=[max(r['validation']['frozen_goal_position_error_m'])*1000 for r in records]
    rot=[np.rad2deg(max(r['validation']['frozen_goal_rotation_error_rad'])) for r in records]
    def m(a):return dict(mean=float(np.mean(a)),median=float(np.median(a)),p95=float(np.percentile(a,95)),worst=float(max(a)))
    return {'total':len(records),'eligible':sum(r['eligible'] for r in records),'position_mm':m(pos),'rotation_deg':m(rot),'endpoint_missing':len(records)-len(pos)}
new=[read_json(p) for p in sorted((OUT/'unguided').glob('*/*.json'))]
old=[read_json(SOURCE/'unguided'/str(sj)/'aligned'/f'{j:02d}.json') for sj in range(2) for j in range(48)]
dev=[read_json(p) for p in sorted((OUT/'comparison/DEV').glob('*/*/candidates/*.json'))]
bad=[r for r in dev if not r['eligible']]
assert all({k for k,v in r['validation']['gates'].items() if not v}=={'geometry_refinement'} for r in bad)
assert all(r['validation']['geometry']['maximum_geometry_step_m']>.002 for r in bad)
assert all(r['validation']['geometry']['refinement_difference_m']<=.001 for r in dev)
result={'status':'PASS','new_native_array_arithmetic':results,'limitations':'Saved full-body geometry/momentum exchange values are inspected and source-traced; no fresh collision queries, mass matrix, robot instantiation or replay is performed. Native low-level joint-range assertions are source/report checked, while planner bounds are independently recomputed.',
    'unguided':{'full323':dist(new),'old_aligned':dist(old)},
    'DEV_rejections':{'count':len(bad),'all_only_geometry_refinement':True,'all_exceed_2mm_density':True,'clearance_refinement_difference_all_below_1mm':True,'bad_density_mm_min':min(r['validation']['geometry']['maximum_geometry_step_m']*1000 for r in bad),'bad_density_mm_max':max(r['validation']['geometry']['maximum_geometry_step_m']*1000 for r in bad)}}
(dest/'native_math_and_tails.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
inputs=read_json(dest/'inputs.json')
for p in (Path(__file__).resolve(),ROOT/'model_test/angle_convention.py'):
    inputs[p.relative_to(ROOT).as_posix()]={'sha256':file_digest(p),'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns}
(dest/'inputs.json').write_text(json.dumps(inputs,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in result.items() if k!='new_native_array_arithmetic'},ensure_ascii=False,indent=2))

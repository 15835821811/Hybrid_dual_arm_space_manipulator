"""Independent saved-array/hash algebra only. No project imports or physics calls."""
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile
from datetime import datetime, timezone

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.interpolate import BSpline

ROOT = Path(__file__).resolve().parents[4]
TRACE = Path(__file__).resolve().parent
RUN = ROOT / 'artifacts/v7_1/diffusion_demo_001'
reviewed = set()

def plain(x):
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, Path): return str(x)
    raise TypeError(type(x).__name__)

def digest(x):
    return hashlib.sha256(json.dumps(x, default=plain, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def sha(p):
    reviewed.add(p.resolve())
    return hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()

def read(p):
    reviewed.add(p.resolve())
    return json.loads(p.read_text(encoding='utf-8-sig'))

def arrays(p):
    reviewed.add(p.resolve())
    with np.load(p, allow_pickle=False) as f: return {k: f[k] for k in f.files}

def maxabs(x): return float(np.max(np.abs(x), initial=0))

run = read(RUN/'run.json'); ident = run['identity']
unit = RUN/'units/u00'; execution = unit/'execution'
n = arrays(execution/'native_trace.npz'); r = arrays(execution/'replay_trace.npz')
init = arrays(execution/'initial_state.npz')['values']; imeta=read(execution/'initial_state.json')
er = read(execution/'report.json'); ledger=read(RUN/'ledger.json'); budget=read(RUN/'budget.json')
task=read(execution/'frozen_task.json'); scene=ident['selection']['entries'][0]
contract=read(RUN/'physical_contract.json'); audit=read(RUN/'scenes/capture_3_obstacles/initial_audit.json')
result=read(unit/'result.json'); rs=read(unit/'repair_selection.json'); es=read(unit/'execution_selection.json')
calls=read(unit/'calls.json'); selection=read(ROOT/'ff_prior_diffusion_demo/evidence/selection.json')
binding=read(ROOT/'ff_prior_diffusion_demo/evidence/binding.json')
software=read(ROOT/'ff_prior_diffusion_demo/evidence/software_tests.json')

checks={}; metrics={}
with zipfile.ZipFile(RUN/'executed_source.zip') as archive:
    archive_names=archive.namelist()
    checks['source_archive_exact_set']=set(archive_names)==set(ident['source_hashes'])
    checks['source_archive_hashes']=all(hashlib.sha256(archive.read(p)).hexdigest()==h for p,h in ident['source_hashes'].items())
    # Compare all frozen files to their exact committed producer bytes. Windows CRLF is recorded separately.
    git_exact=[];git_lf=[];git_mismatch=[]
    for p in ident['source_hashes']:
        frozen=archive.read(p)
        proc=subprocess.run(['git','show',run['producer_head']+':'+p],cwd=ROOT,capture_output=True)
        if proc.returncode: git_mismatch.append(p+':missing')
        elif proc.stdout==frozen: git_exact.append(p)
        elif proc.stdout.replace(b'\r\n',b'\n')==frozen.replace(b'\r\n',b'\n'): git_lf.append(p)
        else: git_mismatch.append(p)
    checks['source_git_commit_content_match']=not git_mismatch
checks['source_current_files_match']=all(sha(ROOT/p)==h for p,h in ident['source_hashes'].items())
checks['binding_files_match']=all(sha(ROOT/p)==h for p,h in ident['binding']['files'].items())
checks['software_test_source_binding']=software['source_hashes']==ident['source_hashes']
checks['selection_and_binding_exact']=selection==ident['selection'] and binding==ident['binding']
checks['producer_head']=run['producer_head']=='8456203e1c0f957f02b6932bc54480897a418146'
checks['ledger_identity']=ledger['identity']['identity_hash']==digest(ident)
checks['no_unknown_work']=ledger['inflight'] is None and not ledger.get('unresolved_call_accounting')
checks['one_attempt_one_unit']=ledger['attempts']=={'static_fullchain':1} and set(ledger['completed'])=={'u00'}
checks['accounting']=all(ledger['calls']['static_fullchain.'+k]==v for k,v in calls.items()) and calls['physics_steps']==27500==budget['actual_native_and_replay_steps'] and calls['coupled']==3932==budget['actual_coupled']==budget['reserved_coupled']
checks['task_unchanged']=task==scene['task'] and task['task_hash']==es['task_hash']==er['task_hash']
checks['task_self_hash']=task['task_hash']==digest({k:v for k,v in task.items() if k!='task_hash'})
checks['original_acceptance']=task['contract']['acceptance']==contract['acceptance']
checks['initial_identity']=digest({'values':init,'model_hash':imeta['model_hash'],'state_mask':imeta['state_mask']})==imeta['state_id']==er['initial_state_id']==task['contract']['initial_state_id']
checks['initial_array_exact']=np.array_equal(init,n['integration'][0]) and np.array_equal(init,r['integration'][0])
checks['scene_identity']=task['contract']['robot_hash']==imeta['model_hash']==er['scene_model_hash']==audit['scene_transfer']['scene_robot_hash'] and task['contract']['pair_policy_hash']==er['pair_policy_hash']==audit['initial_geometry']['pair_policy_hash']
checks['three_static_obstacles']=scene['scene']['motion']=='STATIC' and len(scene['scene']['objects'])==3 and audit['scene_transfer']['added_world_geoms']==3
checks['full_native_shape']=n['integration'].shape==(13751,836) and n['ctrl'].shape==(13750,67) and n['qpos'].shape==(13751,81) and n['qvel'].shape==(13751,79)
checks['all_arrays_finite']=all(np.isfinite(x).all() for x in list(n.values())+list(r.values()))
replay_errors={k:maxabs(n[k]-r[k]) for k in r}
checks['all_full_replay_arrays_exact']=all(v==0 for v in replay_errors.values())
checks['ctrl_hash']=digest(n['ctrl'])==er['ctrl_hash']
checks['native_no_torque_clipping']=np.array_equal(n['ctrl'],n['raw_torque'])
checks['integration_components_match']=np.array_equal(n['integration'][:,0],n['time']) and np.array_equal(n['integration'][:,1:82],n['qpos']) and np.array_equal(n['integration'][:,82:161],n['qvel']) and np.array_equal(n['integration'][:,161:240],n['qacc_warmstart']) and np.array_equal(n['integration'][1:,240:307],n['ctrl'])
checks['integration_no_external_wrench']=not np.any(n['integration'][:,307:836])
t=n['time']-n['time'][0];hold=t>=27-1e-9
checks['complete_time']=len(t)==13751 and maxabs(np.diff(t)-.002)<1e-12 and abs(t[-1]-27.5)<1e-9
goals=np.asarray(task['condition']['goals'])
pos=np.linalg.norm(n['ee'][...,:3,3]-goals[None,:,:3,3],axis=-1)
rel=np.swapaxes(goals[None,:,:3,:3],-1,-2)@n['ee'][...,:3,:3]
rot=Rotation.from_matrix(rel.reshape(-1,3,3)).magnitude().reshape(pos.shape)
metrics.update(final_position_m=pos[-1],final_rotation_rad=rot[-1],hold_max_position_m=np.max(pos[hold],axis=0),hold_max_rotation_rad=np.max(rot[hold],axis=0),hold_frames=int(hold.sum()),time_s=float(t[-1]))
checks['goal_metrics_match']=all(np.allclose(metrics[key],er[rkey],rtol=0,atol=1e-14) for key,rkey in [('final_position_m','final_position_error_m'),('final_rotation_rad','final_rotation_error_rad'),('hold_max_position_m','hold_max_position_error_m'),('hold_max_rotation_rad','hold_max_rotation_error_rad')])
checks['hold_original_gates']=bool(np.all(pos[hold]<=.005) and np.all(rot[hold]<=np.deg2rad(1)))

# Reconstruct only the documented linear 17-to-67 map; do not build a MuJoCo model.
B=np.zeros((67,17));B[:60,:10]=np.kron(np.eye(5),np.array([[0,1],[1,0],[1,0],[0,1],[0,1],[1,0],[1,0],[0,1],[0,1],[1,0],[1,0],[0,1]])/6);B[60:,10:]=np.eye(7)
Bplus=np.linalg.pinv(B);names=contract['identity']['robot']['low_level_joint_names'];mapping=contract['identity']['joint_index_map']
qi=[mapping[name]['qpos'] for name in names];vi=[mapping[name]['dof'] for name in names]
q=n['qpos'][:,qi]@Bplus.T;v=n['qvel'][:,vi]@Bplus.T;acc=np.diff(v,axis=0)/.002
torque_limits=np.r_[np.full(60,4.),np.full(7,80.)];velocity_limits=np.r_[np.full(10,.8),np.full(7,1.2)];acceleration_limits=np.r_[np.full(10,2.5),np.full(7,4.)]
metrics.update(max_raw_torque_limit_ratio=float(np.max(np.abs(n['raw_torque'])/torque_limits)),max_velocity_limit_ratio=float(np.max(np.abs(v)/velocity_limits)),max_acceleration_limit_ratio=float(np.max(np.abs(acc)/acceleration_limits)),tracking_max_rad=maxabs(q[:-1]-n['reference_q']),manifold_max_rad=maxabs(n['qpos'][:,qi]-q@B.T),base_max_translation_m=float(np.max(np.linalg.norm(n['base'][:,:3]-n['base'][0,:3],axis=1))),momentum_peak_linear_angular=np.array([np.max(np.linalg.norm(n['base_generalized_momentum'][:,s],axis=1)) for s in [slice(0,3),slice(3,6)]]))
checks['decoded_positions_match']=np.allclose(q,n['q_actual_17'],rtol=0,atol=1e-14)
checks['motion_limits']=metrics['max_velocity_limit_ratio']<=1 and metrics['max_acceleration_limit_ratio']<=1
checks['torque_limit']=metrics['max_raw_torque_limit_ratio']<=1
checks['manifold_and_tracking']=metrics['manifold_max_rad']<=.02 and metrics['tracking_max_rad']<=.02
limits=read(RUN/'scenes/capture_3_obstacles/limits.json')
checks['planner_position_limits']=bool(np.all(q>=limits['lower']) and np.all(q<=limits['upper']))
checks['momentum_saved_metric']=np.allclose(metrics['momentum_peak_linear_angular'],er['momentum_peak_linear_angular'],rtol=0,atol=1e-14) and bool(np.all(metrics['momentum_peak_linear_angular']<=er['momentum_threshold_linear_angular']))
geo=arrays(execution/'geometry_scans.npz');gjson=read(execution/'geometry_scans.json')
endpoint=[i for i,g in enumerate(gjson) if g['fraction']==0]
checks['all_native_geometry_states']=len(endpoint)==13751 and np.array_equal(geo['qpos'][endpoint],n['qpos'])
checks['all_midpoints_retained']=len(gjson)==27501 and len(gjson)-len(endpoint)==13750
checks['geometry_json_arrays_match']=np.array_equal(geo['minima'],np.array([g['minimum_clearance_m'] for g in gjson]))
checks['native_clearance_trace_match']=np.array_equal(geo['minima'][endpoint],n['clearance_m'])
metrics['minimum_clearance_m']=float(geo['minima'].min());metrics['geometry_checked_states']=len(gjson)
checks['clearance_metric_and_gate']=metrics['minimum_clearance_m']==er['minimum_clearance_m'] and metrics['minimum_clearance_m']>=0
checks['execution_gates_classification']=all(er['gates'].values()) and er['level']==result['execution_level']=='EXECUTION_VERIFIED'

candidates=[]
for k in range(5):
    p=unit/'candidates'/f'{k:02d}';a=arrays(p/'controls.npz');prov=read(p/'provenance.json');o=read(p/'outcome.json');raw=read(p/'raw/screen.json');prepared=read(p/'prepared/screen.json');at=arrays(p/'raw/trace.npz');pt=arrays(p/'prepared/trace.npz')
    checks[f'candidate_{k}_provenance']=prov['network_calls']==16 and prov['sampling_seed']==20264611+k and all(a[x].shape==(19,17) for x in a) and digest(a['network_raw'])==prov['network_raw_hash'] and digest(a['proposal'])==prov['proposal_hash'] and np.array_equal(a['proposal'],a['prepared']) and np.array_equal(at['Z'],a['network_raw']) and np.array_equal(pt['Z'],a['prepared']) and not prov['private_solution_used'] and not prov['fallback_to_other_method']
    item=dict(id=k,seed=prov['sampling_seed'],network_calls=prov['network_calls'],rank=o['rank'],raw_pose_pass=raw['pose_pass_at_screen_grid'],prepared_pose_pass=prepared['pose_pass_at_screen_grid'],prepared_minimum_clearance_m=float(pt['minima'].min()),status=o['status'],eligible=o['eligible'],selected_for_repair=o['selected_for_repair'])
    checks[f'candidate_{k}_rank']=abs(o['rank']-(prepared['metrics']['collision_violation']+.05*np.dot(prepared['goal'],prepared['goal'])))<1e-10
    if o['selected_for_repair']:
        repair=read(p/'repair/result.json');va=read(p/'validated/validation.json');pred=arrays(p/'validated/prediction.npz');g=arrays(p/'validated/grid_2048_geometry_scans.npz')
        item.update(repair_iterations=repair['iterations'],score=va['score'],validated_minimum_clearance_m=float(g['minima'].min()))
        ppos=np.linalg.norm(pred['ee'][-1,:,:3,3]-goals[:,:3,3],axis=-1)
        prot=Rotation.from_matrix(np.swapaxes(goals[:,:3,:3],-1,-2)@pred['ee'][-1,:,:3,:3]).magnitude()
        checks[f'candidate_{k}_qualification']=va['eligible'] and all(va['validation']['gates'].values()) and np.all(ppos<=.005) and np.all(prot<=np.deg2rad(1)) and g['minima'].min()>=0 and np.allclose(ppos,va['validation']['frozen_goal_position_error_m'],rtol=0,atol=1e-14) and np.allclose(prot,va['validation']['frozen_goal_rotation_error_rad'],rtol=0,atol=1e-14)
    candidates.append(item)
checks['candidate_denominator']=len(list((unit/'candidates').iterdir()))==5==result['candidate_denominator'] and result['execution_denominator']==1
checks['rank_selection']=rs['ids']==[x['id'] for x in sorted(candidates,key=lambda x:(x['rank'],x['id']))[:2]]==[4,2]
checks['execution_selection']=es['candidate_id']==sorted([x for x in candidates if x['eligible']],key=lambda x:(x['score'],x['id']))[0]['id']==4 and es['candidate_digest']==digest(result['candidates'])
checks['qualified_count']=sum(x['eligible'] for x in candidates)==2==result['qualified']
chosen=read(unit/'candidates/04/validated/validation.json');final=arrays(unit/'candidates/04/repair/final_path.npz');z=np.asarray(chosen['Z']);q0=np.asarray(task['condition']['initial_features']);knots=np.r_[np.zeros(4),np.arange(1,21)/21,np.ones(4)];control=np.vstack([np.repeat(q0[None],3,axis=0),z[:18],np.repeat(z[-1:],3,axis=0)])
checks['selected_reference_identity']=np.array_equal(z,final['Z']) and digest({'C':control,'knots':knots,'degree':3,'T':27.,'hold':.5})==chosen['reference_hash']==er['metadata']['reference_hash']
curve=BSpline(knots,control,3,extrapolate=False);command_times=np.arange(13750)*.002
for j,key in enumerate(['reference_q','reference_qd','reference_qdd']):
    calculated=curve.derivative(j)(np.clip(command_times/27,0,1))/27**j
    if j: calculated=np.where((command_times>27)[:,None],0,calculated)
    checks['selected_spline_'+key]=maxabs(calculated-n[key])<1e-13

visual=RUN/'visuals';manifest=read(visual/'manifest.json');media={}
checks['media_file_hashes']=all(sha(visual/p)==v['sha256'] and (visual/p).stat().st_size==v['bytes'] for p,v in manifest['files'].items())
checks['media_source_hash']=sha(execution/'native_trace.npz')==manifest['source_native_hash']
checks['media_expected_files']=all((visual/(stem+'.'+ext)).exists() for stem in ['01_summary','02_joint_motion','03_torque_tracking','04_base_manifold','05_world_paths','06_independent_replay','07_world_coordinates'] for ext in ['png','pdf']) and all((visual/f'{v}_{f:03d}.png').exists() for v in ['overview','top'] for f in [0,200,399])
for vname in ['overview','top']:
    proc=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-count_frames','-show_entries','stream=width,height,r_frame_rate,nb_read_frames,duration','-of','json',str(visual/f'{vname}.mp4')],capture_output=True,text=True,check=True)
    media[vname]=json.loads(proc.stdout)['streams'][0]
checks['video_dimensions_frames']=all(v['width']==960 and v['height']==720 and v['nb_read_frames']=='400' and v['r_frame_rate']=='15/1' for v in media.values())
rows=list(csv.DictReader((visual/'timeseries.csv').open(encoding='utf-8',newline='')))
csv_values=np.array([[float(row[k]) for k in ['time_s','arm0_position_mm','arm1_position_mm','arm0_rotation_deg','arm1_rotation_deg','clearance_mm']] for row in rows]);expected=np.column_stack([t,pos*1000,np.rad2deg(rot),n['clearance_m']*1000])
checks['csv_all_times_and_values']=csv_values.shape==(13751,6) and maxabs(csv_values-expected)<1e-10 and np.array_equal([row['hold_gate_active']=='True' for row in rows],hold)
text=(visual/'index.html').read_text(encoding='utf-8');payload=json.loads(re.search(r'const D=(.*?);const c=',text).group(1));indices=np.unique(np.linspace(0,len(t)-1,400).round().astype(int))
checks['html_saved_times_ee_goals']=np.array_equal(payload['time'],t[indices]) and np.array_equal(payload['ee'],n['ee'][indices,:,:3,3]) and np.array_equal(payload['goals'],goals[:,:3,3]) and payload['obstacles']==scene['scene']['objects']
checks['html_saved_errors_clearance']=np.allclose(payload['position_mm'],pos[indices]*1000,rtol=0,atol=1e-10) and np.array_equal(payload['clearance_mm'],n['clearance_m'][indices]*1000)
checks['media_complete_record']=manifest['status']=='COMPLETE' and len(manifest['videos'])==2 and manifest['errors']==[] and manifest['new_native_steps']==0
metrics.update(network_calls_total=sum(x['network_calls'] for x in candidates),native_steps=len(n['ctrl']),replay_steps=len(r['ctrl']),integration_dimensions=n['integration'].shape[1],maximum_geometry_step_m_reported=er['maximum_geometry_step_m'],replay_maximum_errors=replay_errors,source_files=len(ident['source_hashes']),bound_inputs=len(ident['binding']['files']),source_git_exact_files=len(git_exact),source_git_eol_only_files=len(git_lf),source_git_mismatches=git_mismatch,manifest_files=len(manifest['files']),csv_rows=len(rows),html_frames=len(indices),saturation_fraction=float(np.mean(np.abs(n['raw_torque'])>torque_limits)))
nonblocking=['source_git_commit_content_match']
evidence=dict(scope='diffusion_demo_001 only; saved evidence and source inspection; no new physics, optimizer, sampling, or old-run audit',checks=checks,all_checks_pass=all(checks.values()),required_evidence_checks_pass=all(v for k,v in checks.items() if k not in nonblocking),nonblocking_failed_checks=[k for k in nonblocking if not checks[k]],source_provenance_caveat='Producer commit 8456203e plus frozen source_hashes/executed_source.zip, not a pristine Git tree: pre-existing v6_lite/hierarchical_qp.py posture-reference patch is included and hash-bound. Imported via v6_lite/__init__, but this episode uses floating_ff_torque_pd_v1, not HierarchicalVelocityQP.',metrics=metrics,candidates=candidates,video_metadata=media,source_reasoning=dict(model_load='TransferDenoiser loads old aligned model state_dict; rollout calls clean 16 times; recorded calls bound to raw hashes',actual_loop='initial.reset once before loop; loop writes ctrl, then mj_step; no qpos/qvel/state reset',independent_replay='new initial.reset once, saved ctrl replayed through mj_step; never copies recorded qpos/qvel inside replay loop',geometry='saved 27501 scans retain every native state plus midpoint; 2mm geometry extent is source/report evidence, not independently rerun',visuals='separate MjData, saved qpos/qvel and mj_forward only; 400 existing-state indices; no mj_step'),paper_claim_audit='unavailable',downstream_writing='provisional — no paper claim audit available',pending_external_review=False)
(TRACE/'compact_evidence.json').write_text(json.dumps(evidence,default=plain,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
for p in RUN.rglob('*'):
    if p.is_file(): reviewed.add(p.resolve())
reviewed.add((ROOT/'ff_prior_diffusion_demo/docs/EXPERIMENT_PLAN.md').resolve())
inputs=[]
for p in sorted(reviewed):
    inputs.append(dict(path=str(p),sha256=hashlib.file_digest(p.open('rb'),'sha256').hexdigest(),bytes=p.stat().st_size,mtime_utc=datetime.fromtimestamp(p.stat().st_mtime,timezone.utc).isoformat()))
(TRACE/'inputs.json').write_text(json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(),producer_head=run['producer_head'],reviewed_artifacts=inputs),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(checks=len(checks),all_checks_pass=all(checks.values()),failed=[k for k,v in checks.items() if not v],metrics=metrics,media=media),default=plain,ensure_ascii=True))

"""Bounded read-only array/source audit: no MuJoCo import, step, optimization or old audit."""
import sys
sys.dont_write_bytecode = True
import json, hashlib, zipfile, subprocess, datetime, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from model_test.angle_convention import continuum_mujoco_map_60_to_10
from v6_lite.irregular_waypoints import IrregularWaypointTarget
OUT = Path(__file__).resolve().parent
RUN = ROOT/'artifacts/v7_1/terminal_tracking_001'
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()
def plain(v):
    if isinstance(v,np.ndarray): return v.tolist()
    if isinstance(v,np.generic): return v.item()
    raise TypeError(type(v).__name__)
def digest(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def save(name, data): (OUT/name).write_text(json.dumps(data,default=plain,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
ident=read(RUN/'run.json'); ident0=ident['identity']; c=ident0['contract']; p=ident0['protocol']
report=read(RUN/'execution/result.json'); budget=read(RUN/'budget.json'); media=read(RUN/'visuals/manifest.json')
inputs={}
for rel in set(ident0['source_hashes']) | set(ident0['binding']['files']) | {
    str(f.relative_to(ROOT)).replace('\\','/') for f in RUN.rglob('*') if f.is_file()} | {
    'ff_prior_diffusion_demo/evidence/selection.json'}:
    f=ROOT/rel; st=f.stat(); inputs[rel]={'path':str(f.resolve()),'sha256':sha(f),'bytes':st.st_size,
        'mtime_utc':datetime.datetime.fromtimestamp(st.st_mtime,datetime.timezone.utc).isoformat()}
source_bad=[f for f,h in ident0['source_hashes'].items() if inputs[f]['sha256']!=h]
binding_bad=[f for f,h in ident0['binding']['files'].items() if inputs[f]['sha256']!=h]
with zipfile.ZipFile(RUN/'executed_source.zip') as z:
    archive_bad=[f for f,h in ident0['source_hashes'].items() if hashlib.sha256(z.read(f)).hexdigest()!=h]
    archive_files=len(z.namelist())
git_bad=[]; git_missing=[]; git_eol=[]
for f,h in ident0['source_hashes'].items():
    q=subprocess.run(['git','show',ident['producer_head']+':'+f],cwd=ROOT,capture_output=True)
    if q.returncode: git_missing.append(f)
    elif hashlib.sha256(q.stdout).hexdigest()!=h:
        if q.stdout.replace(b'\r\n',b'\n')==(ROOT/f).read_bytes().replace(b'\r\n',b'\n'):git_eol.append(f)
        else:git_bad.append(f)
cc=dict(c); ch=cc.pop('contract_hash')
source=ROOT/'artifacts/v7_1/diffusion_demo_001/units/u00/execution/native_trace.npz'
with np.load(source) as z:
    last={k:z[k][-1].copy() for k in ('integration','qpos','qvel','time','reference_q','reference_qd')}
    source_rows=len(z['time'])
with np.load(ROOT/'ff_prior_terminal_tracking/evidence/terminal.npz') as z: terminal=dict(z)
with np.load(RUN/'initial.npz') as z: initial=z['integration']
with np.load(RUN/'execution/native_trace.npz') as z: a=dict(z)
with np.load(RUN/'execution/replay_trace.npz') as z: b=dict(z)
with np.load(RUN/'execution/geometry.npz') as z: g=dict(z)
checks={}
checks['identity']={'producer_head':ident['producer_head'],'source_count':len(ident0['source_hashes']),
    'input_count':len(ident0['binding']['files']),'source_current_mismatches':source_bad,'input_mismatches':binding_bad,
    'source_archive_mismatches':archive_bad,'archive_file_count':archive_files,
    'git_blob_differences':git_bad,'git_blob_missing':git_missing,'git_line_ending_only_differences':git_eol,
    'contract_digest_valid':digest(cc)==ch,'contract_matches_file':c==read(ROOT/'ff_prior_terminal_tracking/evidence/contract.json'),
    'source_native_hash_matches':sha(source)==c['source_native_sha256']}
checks['handoff']={'predecessor_rows':source_rows,'predecessor_last_row_index':source_rows-1,
    'terminal_fields_equal_last_source_row':{k:np.array_equal(last[k],terminal[k]) for k in last},
    'full836_source_terminal_initial_native_replay_equal':all(np.array_equal(last['integration'],x) for x in (terminal['integration'],initial,a['integration'][0],b['integration'][0])),
    'command_q_equal':np.array_equal(a['qref'][0],last['reference_q']),
    'time_start':a['time'][0],'time_end':a['time'][-1],
    'time_step_max_error':np.max(abs(np.diff(a['time'])-.002)),
    'time_duration':a['time'][-1]-a['time'][0]}
checks['replay']={k:float(np.max(abs(a[k]-b[k]))) for k in ('integration','qpos','qvel','ctrl')}
checks['state_consistency']={'time':np.array_equal(a['time'],a['integration'][:,0]),
    'qpos':np.array_equal(a['qpos'],a['integration'][:,1:82]),
    'qvel':np.array_equal(a['qvel'],a['integration'][:,82:161]),
    'ctrl_after_step':np.array_equal(a['ctrl'],a['integration'][1:,240:307]),
    'applied_force_and_remaining_state_max_abs':np.max(abs(a['integration'][:,307:])),
    'target_qvel_max_abs':np.max(abs(a['qvel'][:,73:79])),
    'target_qpos_drift_max':np.max(abs(a['qpos'][:,74:81]-a['qpos'][0,74:81])),
    'all_saved_arrays_finite':all(np.isfinite(x).all() for x in (*a.values(),*b.values(),*g.values()))}
old=read(ROOT/'artifacts/v7_1/dynamic_handoff_001/u00/task.json')['waypoints']; w=c['waypoints']
goals=np.asarray(c['static_goals']); selected=read(ROOT/'ff_prior_diffusion_demo/evidence/selection.json')['entries'][0]
target=IrregularWaypointTarget(np.asarray(w['initial_position_w']),np.asarray(w['waypoint_points_m']),
    np.asarray(w['target_rotation_world']),w['transition_duration_s'],w['path_duration_s'],np.asarray(w['segment_durations_s']),w)
local=a['time']-a['time'][0]
desired_pos=np.asarray([target.sample(t)[0] for t in local])
rv=Rotation.from_matrix(np.asarray(w['target_rotation_world'])@goals[0,:3,:3].T).as_rotvec()
s=np.clip(local/4.5,0,1); alpha=s**3*(10-15*s+6*s*s)
desired_rot=Rotation.from_rotvec(alpha[:,None]*rv).as_matrix()@goals[0,:3,:3]
checks['targets']={'waypoint_changed_keys':[k for k in old if old[k]!=w[k]],
    'all_seven_points_exact':old['waypoint_points_m']==w['waypoint_points_m'],
    'timing_exact':all(old[k]==w[k] for k in ('transition_duration_s','path_duration_s','segment_durations_s','path_start_s','path_end_s')),
    'static_goals_equal_predecessor_designated':np.array_equal(goals,np.asarray(selected['task']['condition']['goals'])),
    'bridge_start_equal_designated_goal':np.array_equal(np.asarray(w['initial_position_w']),goals[0,:3,3]),
    'all_native_desired_position_max_error':np.max(abs(a['desired'][:,0,:3,3]-desired_pos)),
    'all_native_desired_rotation_max_error':np.max(abs(a['desired'][:,0,:3,:3]-desired_rot)),
    'rigid_goal_stays_designated_max_error':np.max(abs(a['desired'][:,1]-goals[1])),
    'waypoint_event_local_times':np.r_[4.5,4.5+np.cumsum(w['segment_durations_s'])],
    'waypoint_events_exact_algebra_max_error':np.max(abs(np.asarray([target.sample(t)[0] for t in np.r_[4.5,4.5+np.cumsum(w['segment_durations_s'])]])-np.asarray(w['waypoint_points_m'])))}
pos=np.linalg.norm(a['ee'][:,:,:3,3]-a['desired'][:,:,:3,3],axis=-1)
relative=a['desired'][:,:,:3,:3]@np.swapaxes(a['ee'][:,:,:3,:3],-1,-2)
rot=Rotation.from_matrix(relative.reshape(-1,3,3)).magnitude().reshape(-1,2)
def stat(x):return {'maximum':x.max(axis=0),'mean':x.mean(axis=0),'p95':np.percentile(x,95,axis=0),'final':x[-1]}
B=np.zeros((67,17)); B[:60,:10]=continuum_mujoco_map_60_to_10();B[60:,10:]=np.eye(7);Binv=np.linalg.pinv(B)
lowq=a['qpos'][:,7:74];lowv=a['qvel'][:,6:73];vel=lowv@Binv.T
vl=np.r_[np.full(10,.8),np.full(7,1.2)];al=np.r_[np.full(10,2.5),np.full(7,4.)];tl=np.r_[np.full(60,4.),np.full(7,80.)]
lo=np.r_[np.full(10,-1.),np.full(7,-np.pi)];hi=-lo
tree=ET.parse(ROOT/'dual_arm_space_robot_2026/urdf/dual_arm_space_robot_2026.urdf')
joints=[j for j in tree.findall('joint') if j.attrib['type'] in ('revolute','continuous','prismatic')]
limits=np.asarray([(float(j.find('limit').attrib['lower']),float(j.find('limit').attrib['upper']))
    if j.attrib['type']!='continuous' and j.find('limit') is not None else (-np.inf,np.inf) for j in joints])
filter_rows=read(RUN/'control_filter.json'); filtered=np.asarray([row['filtered'] for row in filter_rows]);before=np.vstack([last['reference_qd'],filtered[:-1]])
qd=(before[:,None,:]+np.arange(1,11)[None,:,None]/10*(filtered-before)[:,None,:]).reshape(-1,17)
qdd=np.repeat((filtered-before)/.02,10,axis=0)
mp=np.array([np.max(np.linalg.norm(a['momentum'][:,s],axis=1)) for s in (slice(0,3),slice(3,6))])
me=np.array([np.max(np.linalg.norm(a['exchange'][:,s],axis=1)) for s in (slice(0,3),slice(3,6))])
checks['metrics']={'position_mm':stat(pos*1000),'rotation_deg':stat(np.rad2deg(rot)),
    'minimum_clearance_mm':1000*np.min(g['minima']), 'raw_torque_ratio_max':np.max(abs(a['raw_torque'])/tl),
    'actual_velocity_ratio_max':np.max(abs(vel)/vl),'actual_acceleration_ratio_max':np.max(abs(np.diff(vel,axis=0)/.002)/al),
    'low_level_velocity_ratio_max':np.max(abs(lowv)/(abs(B)@vl)),
    'low_level_acceleration_ratio_max':np.max(abs(np.diff(lowv,axis=0)/.002)/(abs(B)@al)),
    'command_velocity_ratio_max':np.max(abs(a['reference_qd'])/vl),'command_acceleration_ratio_max':np.max(abs(a['reference_qdd'])/al),
    'joint_actual_margin_rad':min(np.min(a['q17']-lo),np.min(hi-a['q17'])),
    'joint_command_margin_rad':min(np.min(a['qref']-lo),np.min(hi-a['qref'])),
    'low_joint_margin_rad':min(np.min(lowq-limits[:,0]),np.min(limits[:,1]-lowq)),
    'tracking_max_rad':np.max(abs(a['q17']-a['qref'])), 'manifold_max_rad':np.max(abs(lowq-a['q17']@B.T)),
    'momentum_peak':mp,'momentum_threshold':1e-5+.002*me,
    'base_displacement_max_mm':1000*np.max(np.linalg.norm(a['base'][:,:3]-a['base'][0,:3],axis=1))}
checks['command']={'raw_torque_vs_ctrl_max_error':np.max(abs(a['raw_torque']-a['ctrl'])),
    'ramp_qd_from_inherited_previous_command_max_error':np.max(abs(qd-a['reference_qd'])),
    'ramp_qdd_from_inherited_previous_command_max_error':np.max(abs(qdd-a['reference_qdd'])),
    'qref_integration_max_error':np.max(abs(a['qref'][1:]-a['qref'][:-1]-.002*a['reference_qd'])),
    'q17_from_qpos_max_error':np.max(abs(a['q17']-lowq@Binv.T)), 'control_ticks':len(filter_rows)}
clear=np.r_[g['minima'][0],np.minimum(g['minima'][1::2],g['minima'][2::2])]
checks['geometry']={'rows':len(g['qpos']),'native_rows_exact':np.array_equal(g['qpos'][1::2],a['qpos'][1:]),
    'first_row_exact':np.array_equal(g['qpos'][0],a['qpos'][0]),
    'midpoint_scalar_max_error':np.max(abs(g['qpos'][2::2,7:74]-(lowq[:-1]+lowq[1:])/2)),
    'minima_reduction_equals_trace':np.array_equal(clear,a['clearance']),
    'recorded_max_geometry_substep_m':report['maximum_geometry_step_m'],
    'recorded_substep_below_gate':report['maximum_geometry_step_m']<=.002,
    'geometry_pair_rows':len(read(RUN/'execution/geometry_pairs.json')),
    'distance_recomputation_performed':False,'continuous_time_certificate':False}
csv=np.genfromtxt(RUN/'visuals/timeseries.csv',delimiter=',',skip_header=1)
expected_csv=np.column_stack([a['time'],local,pos*1000,np.rad2deg(rot),a['clearance']*1000])
checks['media']={'manifest_files_mismatch':[n for n,h in media['files'].items() if sha(RUN/'visuals'/n)!=h['sha256']],
    'csv_rows':len(csv),'csv_vs_arrays_max_error':np.max(abs(csv-expected_csv)),
    'dynamic_source_hash_matches':media['source_dynamic_hash']==sha(RUN/'execution/native_trace.npz'),
    'static_source_hash_matches':media['source_static_hash']==c['source_native_sha256'],
    'recorded_duration_s':media['combined_actual_duration_s'],'videos':media['videos'],
    'native_steps_for_visualization':media['native_steps_for_visualization']}
events=[json.loads(s) for s in (RUN/'events.jsonl').read_text().splitlines()]
checks['budget']={'native_arrays':len(a['ctrl']),'replay_arrays':len(b['ctrl']),
    'ledger':budget,'event_count':len(events),'status':read(RUN/'status.json'),
    'output_bytes':sum(f.stat().st_size for f in RUN.rglob('*') if f.is_file()),
    'new_physics_or_forward_in_review':0,'comparison_runs':0,'training_updates':0}
checks['gate_agreement']={'reported_gates':report['gates'],
    'position_stats_max_error':max(float(np.max(abs(np.asarray(report['position_mm'][k])-v))) for k,v in checks['metrics']['position_mm'].items()),
    'rotation_stats_max_error':max(float(np.max(abs(np.asarray(report['rotation_deg'][k])-v))) for k,v in checks['metrics']['rotation_deg'].items()),
    'paper_claim_audit':'unavailable' if not (ROOT/'paper/PAPER_CLAIM_AUDIT.json').exists() else read(ROOT/'paper/PAPER_CLAIM_AUDIT.json')}
save('evidence.json',checks)
save('inputs.json',{'reviewed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'artifacts':inputs,
    'scope':'D2 only; predecessor terminal row and bound hashes; saved-data review; no new simulation'})
print(json.dumps(checks,default=plain,ensure_ascii=True,indent=2))

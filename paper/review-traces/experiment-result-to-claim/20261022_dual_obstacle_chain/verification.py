"""Saved-evidence review only. No MuJoCo model construction, stepping, FK or distance calls.
Run from project root with .venv/Scripts/python.exe and PYTHONDONTWRITEBYTECODE=1.
Only this review directory is written. Existing physics/replay/geometry evidence is inspected.
"""
from pathlib import Path
import sys, json, hashlib, zipfile, datetime, re, xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
RUN=ROOT/'artifacts/v7_1/dual_obstacle_chain_001'
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT))
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def plain(v):
 if isinstance(v,np.ndarray):return v.tolist()
 if isinstance(v,np.generic):return v.item()
 raise TypeError(type(v).__name__)
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),default=plain,allow_nan=False).encode()).hexdigest()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def write(name,v):(OUT/name).write_text(json.dumps(v,indent=2,ensure_ascii=False,default=plain,allow_nan=False)+'\n',encoding='utf-8')
def npz(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def maximum(a):return float(np.max(np.abs(a),initial=0))
def pose(actual,desired):
 p=np.linalg.norm(actual[...,:3,3]-desired[...,:3,3],axis=-1)
 R=np.swapaxes(desired[...,:3,:3],-1,-2)@actual[...,:3,:3]
 rot=Rotation.from_matrix(R.reshape(-1,3,3)).magnitude().reshape(p.shape)
 return p,rot
def dist(x):return dict(mean=np.mean(x,axis=0),p95=np.percentile(x,95,axis=0),maximum=np.max(x,axis=0),final=x[-1])
checks={}
def check(name,value):
 checks[name]=bool(value)
 return bool(value)
run=read(RUN/'run.json');ident=run['identity'];contract=ident['contract'];selection=ident['selection'];task=selection['entries'][0]['task'];P=ident['protocol']
source=zipfile.ZipFile(RUN/'executed_source.zip')
source_mismatch=[];working_mismatch=[];inputs_mismatch=[];inventory=[]
for p,h in ident['source_hashes'].items():
 if hashlib.sha256(source.read(p)).hexdigest()!=h:source_mismatch.append(p)
 if sha(ROOT/p)!=h:working_mismatch.append(p)
for p,h in ident['binding']['files'].items():
 if sha(ROOT/p)!=h:inputs_mismatch.append(p)
for group,paths in [('run',sorted(RUN.rglob('*'))),('source',[ROOT/p for p in ident['source_hashes']]),('frozen_input',[ROOT/p for p in ident['binding']['files']])]:
 for p in paths:
  if p.is_file():inventory.append(dict(group=group,path=p.relative_to(ROOT).as_posix(),sha256=sha(p),bytes=p.stat().st_size,mtime_utc=datetime.datetime.fromtimestamp(p.stat().st_mtime,datetime.timezone.utc).isoformat()))
check('source_archive_208_entries_match',len(ident['source_hashes'])==208 and not source_mismatch and set(source.namelist())==set(ident['source_hashes']))
check('working_sources_match_archive',not working_mismatch)
check('frozen_93_inputs_match',len(ident['binding']['files'])==93 and not inputs_mismatch)
check('producer_head',run['producer_head']=='46b552fded439dce19a85b02430159e18508ee89')
check('task_hash',digest({k:v for k,v in task.items() if k!='task_hash'})==task['task_hash'])
check('contract_hash',digest({k:v for k,v in contract.items() if k!='contract_hash'})==contract['contract_hash'])
sp=RUN/'static/units/u00';sn=npz(sp/'execution/native_trace.npz');sr=npz(sp/'execution/replay_trace.npz')
dn=npz(RUN/'dynamic/execution/native_trace.npz');dr=npz(RUN/'dynamic/execution/replay_trace.npz');di=npz(RUN/'dynamic/initial.npz')
sreport=read(sp/'execution/report.json');dreport=read(RUN/'dynamic/execution/result.json');summary=read(RUN/'summary.json')
static_replay={k:maximum(sn[k]-sr[k]) for k in sr}
dynamic_replay={k:maximum(dn[k]-dr[k]) for k in dr}
check('static_full_replay_arrays_exact',all(v==0 for v in static_replay.values()))
check('dynamic_full_replay_arrays_exact',all(v==0 for v in dynamic_replay.values()))
check('raw_shapes',sn['integration'].shape==(13751,836) and dn['integration'].shape==(13501,836) and sn['ctrl'].shape==(13750,67) and dn['ctrl'].shape==(13500,67))
handoff={k:maximum(di[k]-sn[k][-1]) for k in di}
handoff.update({f'dynamic_first_{k}':maximum(dn[k][0]-sn[k][-1]) for k in ['integration','qpos','qvel','time','ee','base']})
handoff['dynamic_first_qref']=maximum(dn['qref'][0]-sn['reference_q'][-1])
check('full836_handoff_and_command_exact',all(v==0 for v in handoff.values()))
check('absolute_clock_native_grids',maximum(np.diff(sn['time'])-.002)<1e-10 and maximum(np.diff(dn['time'])-.002)<1e-10 and abs(sn['time'][-1]-27.5)<1e-9 and abs(dn['time'][-1]-54.5)<1e-9)
check('finite_all_saved_native',all(np.isfinite(v).all() for a in [sn,dn] for v in a.values()))
check('model_identity_crosslinks',sreport['scene_model_hash']==contract['scene_model_hash']==task['contract']['robot_hash']==read(RUN/'dynamic/contract.json')['scene_model_hash'])
check('pair_policy_crosslinks',sreport['pair_policy_hash']==task['contract']['pair_policy_hash']==read(RUN/'initial_scene_audit.json')['initial_geometry']['pair_policy_hash'])
initial=npz(sp/'execution/initial_state.npz')['values'];initialmeta=read(sp/'execution/initial_state.json')
check('initial_raw_exact',np.array_equal(initial,sn['integration'][0]))
check('initial_state_id',digest(dict(values=initial,model_hash=contract['scene_model_hash'],state_mask=initialmeta['state_mask']))==sreport['initial_state_id']==task['contract']['initial_state_id'])
# The saved physical contract gives the unchanged 17->67 mapping index identities.
phys=read(ROOT/'artifacts/v7_1/sample_alignment_001/physical_contract.json');rid=phys['identity']
names=rid['robot']['low_level_joint_names'];maps=rid['joint_index_map'];qi=[maps[n]['qpos'] for n in names];vi=[maps[n]['dof'] for n in names]
from model_test.angle_convention import continuum_mujoco_map_60_to_10
B=np.zeros((67,17));B[:60,:10]=continuum_mujoco_map_60_to_10();B[60:,10:]=np.eye(7);BP=np.linalg.pinv(B)
lower=np.array(rid['planner_lower']);upper=np.array(rid['planner_upper']);vmax=np.r_[np.full(10,.8),np.full(7,1.2)];amax=np.r_[np.full(10,2.5),np.full(7,4.)];tau=np.r_[np.full(60,4.),np.full(7,80.)]
urdf=ET.parse(ROOT/'dual_arm_space_robot_2026/urdf/dual_arm_space_robot_2026.urdf').getroot();joints={j.attrib['name']:j for j in urdf.findall('joint')}
limited=[];ranges=[]
for n in names:
 j=joints[n];lim=j.find('limit');limited.append(j.attrib['type']!='continuous' and lim is not None and 'lower' in lim.attrib and 'upper' in lim.attrib);ranges.append([float(lim.attrib.get('lower','0')),float(lim.attrib.get('upper','0'))] if lim is not None else [0.,0.])
limited=np.array(limited);ranges=np.array(ranges)
motion={}
for label,a,qkey,rkey in [('static',sn,'q_actual_17','reference_q'),('dynamic',dn,'q17','qref')]:
 q=a['qpos'][:,qi];vel=a['qvel'][:,vi];v=vel@BP.T;acc=np.diff(v,axis=0)/.002;lacc=np.diff(vel,axis=0)/.002
 aq=q@BP.T;ref=a[rkey]
 measure=dict(actual_q_decode_error=maximum(aq-a[qkey]),joint_domain_pass=bool(np.all(aq>=lower) and np.all(aq<=upper) and np.all(ref>=lower) and np.all(ref<=upper) and np.all(q[:,limited]>=ranges[limited,0]) and np.all(q[:,limited]<=ranges[limited,1])),velocity_ratio=float(np.max(abs(v)/vmax)),acceleration_ratio=float(np.max(abs(acc)/amax)),low_velocity_ratio=float(np.max(abs(vel)/(abs(B)@vmax))),low_acceleration_ratio=float(np.max(abs(lacc)/(abs(B)@amax))),reference_velocity_ratio=float(np.max(abs(a['reference_qd'])/vmax)),reference_acceleration_ratio=float(np.max(abs(a['reference_qdd'])/amax)),raw_torque_ratio=float(np.max(abs(a['raw_torque'])/tau)),ctrl_raw_max_error=maximum(a['ctrl']-a['raw_torque']),manifold_max_rad=maximum(q-aq@B.T))
 check(label+'_saved_joint_domain',measure['joint_domain_pass'])
 check(label+'_saved_derivative_limits',all(measure[k]<=1 for k in ['velocity_ratio','acceleration_ratio','low_velocity_ratio','low_acceleration_ratio','reference_velocity_ratio','reference_acceleration_ratio']))
 check(label+'_torque_control',measure['raw_torque_ratio']<=1 and measure['ctrl_raw_max_error']==0)
 motion[label]=measure
from ff_prior.spline import Reference
chosen=read(sp/'execution_selection.json')['candidate_id'];validation=read(sp/f'candidates/{chosen:02d}/validated/validation.json');ref=Reference(np.array(validation['Z']),task['condition']['initial_features'],task['condition']['duration'])
ref_q=ref.evaluate(sn['time']-sn['time'][0]);motion['static']['tracking_max_rad']=maximum(sn['q_actual_17']-ref_q)
motion['dynamic']['tracking_max_rad']=maximum(dn['q17']-dn['qref'])
check('static_reference_hash_and_saved_commands',ref.reference_hash==sreport['metadata']['reference_hash'] and maximum(ref.evaluate(np.arange(13750)*.002)-sn['reference_q'])==0)
check('tracking_and_manifold',all(v['tracking_max_rad']<=.02 and v['manifold_max_rad']<=.02 for v in motion.values()))
spose,srot=pose(sn['ee'],np.array(task['condition']['goals']));dpose,drot=pose(dn['ee'],dn['desired'])
shold=sn['time']-sn['time'][0]>=27-1e-9;dhold=dn['time']-dn['time'][0]>=25.5-1e-9
check('static_goal_hold',np.all(spose[shold]<=.005) and np.all(srot[shold]<=np.pi/180))
check('dynamic_pose_all_samples_and_hold',np.all(dpose<=.005) and np.all(drot<=np.pi/180) and dhold.any())
check('dynamic_summary_metrics_match',maximum(np.max(dpose,axis=0)*1000-dreport['position_mm']['maximum'])<1e-12 and maximum(np.max(drot,axis=0)*180/np.pi-dreport['rotation_deg']['maximum'])<1e-12)
sg=npz(sp/'execution/geometry_scans.npz');dg=npz(RUN/'dynamic/execution/geometry.npz')
check('saved_geometry_nonpenetration',np.min(sg['minima'])>=0 and np.min(dg['minima'])>=0)
check('saved_static_geometry_endpoint_and_minima',np.array_equal(sg['qpos'][np.r_[0,np.arange(1,len(sg['qpos']),2)]],sn['qpos']) and np.array_equal(sg['minima'][np.r_[0,np.arange(1,len(sg['qpos']),2)]],sn['clearance_m']) and np.min(sg['minima'])==sreport['minimum_clearance_m'])
dend=np.r_[0,np.arange(1,len(dg['qpos']),2)]
dmin=np.r_[dg['minima'][0],np.min(dg['minima'][1:].reshape(-1,2),axis=1)]
check('saved_dynamic_geometry_endpoint_and_minima',np.array_equal(dg['qpos'][dend],dn['qpos']) and np.array_equal(dmin,dn['clearance']) and np.min(dmin)*1000==dreport['minimum_clearance_mm'])
# INTEGRATION order: time,qpos,qvel,act,warmstart,ctrl,qfrc_applied,xfrc_applied,then remaining user fields.
# Model dimensions and the archive establish no act/equality/mocap/userdata state here.
external={}
for label,a in [('static',sn),('dynamic',dn)]:
 integ=a['integration'];external[label]=dict(time_error=maximum(integ[:,0]-a['time']),qpos_error=maximum(integ[:,1:82]-a['qpos']),qvel_error=maximum(integ[:,82:161]-a['qvel']),external_tail_max=maximum(integ[:,307:]),previous_ctrl_error=maximum(integ[1:,240:307]-a['ctrl']))
check('integration_raw_layout_and_external_wrench_zero',all(all(x==0 for x in v.values()) for v in external.values()))
mp=np.array([np.max(np.linalg.norm(dn['momentum'][:,s],axis=1)) for s in (slice(0,3),slice(3,6))]);me=np.array([np.max(np.linalg.norm(dn['exchange'][:,s],axis=1)) for s in (slice(0,3),slice(3,6))]);smp=np.array([np.max(np.linalg.norm(sn['base_generalized_momentum'][:,s],axis=1)) for s in (slice(0,3),slice(3,6))])
check('saved_momentum_gates',np.all(mp<=1e-5+.002*me) and np.all(smp<=sreport['momentum_threshold_linear_angular']))
check('reported_15_and_19_gates',len(sreport['gates'])==15 and len(dreport['gates'])==19 and all(sreport['gates'].values()) and all(dreport['gates'].values()))
rows=[]
for k in range(5):
 p=sp/f'candidates/{k:02d}';a=npz(p/'controls.npz');prov=read(p/'provenance.json');outcome=read(p/'outcome.json');item=dict(id=k,seed=prov['sampling_seed'],network_calls=prov['network_calls'],outcome=outcome,raw_digest_matches=digest(a['network_raw'])==prov['network_raw_hash'],proposal_digest_matches=digest(a['proposal'])==prov['proposal_hash'])
 if (p/'validated/validation.json').exists():
  val=read(p/'validated/validation.json');item['model_grade']=val['validation']['validation_level'];item['model_gates']=val['validation']['gates'];item['repair_iterations']=len(list((p/'repair/iterations').glob('*/record.json')))
 rows.append(item)
check('five_proposals_provenance',all(r['network_calls']==16 and r['seed']==20264611+r['id'] and r['raw_digest_matches'] and r['proposal_digest_matches'] for r in rows))
check('two_model_valid',sum(r.get('model_grade')=='MODEL_VALID' and all(r['model_gates'].values()) for r in rows)==2)
check('preselected_repair_and_execution',read(sp/'repair_selection.json')['ids']==[4,2] and chosen==min([r for r in rows if r['outcome'].get('eligible')],key=lambda r:(r['outcome']['score'],r['id']))['id']==4)
# Algebraic reference reconstruction uses the saved original exact waypoint contract.
w=contract['waypoints'];points=np.array(w['waypoint_points_m']);dur=np.array(w['segment_durations_s']);bounds=np.r_[4.5,4.5+np.cumsum(dur)];centers=np.array([x['position'] for x in contract['scene']['objects'][3:]])
original=read(ROOT/'artifacts/v7_1/dynamic_handoff_001/u00/task.json')['waypoints']
oldselection=read(ROOT/'ff_prior_diffusion_demo/evidence/selection.json')
check('seven_original_points_timing_rotation_exact',all(w[k]==original[k] for k in ['waypoint_points_m','segment_durations_s','transition_duration_s','path_duration_s','target_rotation_world']) and digest(points)==contract['original_waypoints_sha256'])
check('old_three_and_new_six',len(contract['scene']['objects'])==9 and contract['scene']['objects'][:3]==oldselection['entries'][0]['scene']['objects'] and np.array_equal(centers,(points[:-1]+points[1:])/2) and all(o['shape']=='sphere' and o['size']==[.012,0.,0.] for o in contract['scene']['objects'][3:]))
from v6_lite.irregular_waypoints import IrregularWaypointTarget
from ff_prior_dual_obstacle_chain.path import BypassPath
path=IrregularWaypointTarget(np.array(w['initial_position_w']),points,np.array(w['target_rotation_world']),4.5,21.,dur,w);bypass=BypassPath(path,contract['bypass_offset_m'])
local=dn['time']-dn['time'][0];computed=np.array([bypass.sample(t)[0] for t in local])
check('saved_dynamic_reference_equals_predefined_C2',maximum(computed-dn['desired'][:,0,:3,3])<1e-14)
check('seven_reference_waypoints_exact',maximum(np.array([bypass.sample(t)[0] for t in bounds])-points)<1e-14)
waypoints=[]
for i,t in enumerate(bounds):
 k=int(np.argmin(abs(local-t)));waypoints.append(dict(waypoint=i+1,world_point_m=points[i],scheduled_local_s=float(t),scheduled_absolute_s=float(dn['time'][0]+t),sample_index=k,nearest_local_s=float(local[k]),nearest_absolute_s=float(dn['time'][k]),time_offset_ms=float((local[k]-t)*1000),actual_world_position_m=dn['ee'][k,0,:3,3],point_error_mm=float(np.linalg.norm(dn['ee'][k,0,:3,3]-points[i])*1000),desired_point_offset_mm=float(np.linalg.norm(dn['desired'][k,0,:3,3]-points[i])*1000),rotation_error_deg=float(drot[k,0]*180/np.pi),evidence='nearest saved 2ms sample; no continuous-time arrival claim'))
midtimes=(bounds[:-1]+bounds[1:])/2;oldmid=np.array([path.sample(t)[0] for t in midtimes]);newmid=np.array([bypass.sample(t)[0] for t in midtimes])
obstacle_relevance=dict(original_midpoint_center_error_m=np.linalg.norm(oldmid-centers,axis=1),original_reference_signed_sphere_distance_m=np.linalg.norm(oldmid-centers,axis=1)-.012,bypass_midpoint_offset_m=newmid-centers,bypass_reference_midpoint_signed_sphere_distance_m=np.linalg.norm(newmid-centers,axis=1)-.012,scope='coordinate algebra only; not a new collision/geometry certification')
check('six_obstacles_block_original_reference_midpoints',maximum(oldmid-centers)<1e-12)
html=(RUN/'dynamic/visuals/index.html').read_text(encoding='utf-8');match=re.search(r'const D=(.*?);',html)
if match is None:match=re.search(r'const D = (.*?);',html)
payload=json.loads(match.group(1)) if match else None
check('html_contains_same_nine_obstacles',payload is not None and payload['obstacles']==contract['scene']['objects'])
transitions=read(RUN/'dynamic/execution/transitions.json');trans=[]
for tr in transitions:
 z=npz(RUN/f"dynamic/execution/transition_{int(tr['time']*10)}.npz");k=tr['index'];trans.append(all(np.array_equal(z[key],dn[key][k-1:k+2]) for key in z))
check('transition_triplets_raw_exact',len(trans)==2 and all(trans))
sb=read(RUN/'static/budget.json');db=read(RUN/'dynamic/budget.json');calls=read(RUN/'static/calls.json');status=read(RUN/'status.json');events=[json.loads(x) for x in (RUN/'events.jsonl').read_text().splitlines() if x.strip()]
eventstages={name:[e for e in events if e['stage']==name] for name in ['public_proposals','native67_execution_and_independent_replay','actual_terminal_handoff_same_nine_obstacles','nine_obstacle_tracking','independent_replay']}
check('observed_static_counts_and_no_replacement',calls['physics_steps']==2*len(sn['ctrl'])==27500 and calls['torque_rollout']==1 and sb['actual_native_and_replay_steps']==27500 and calls['coupled']==sb['actual_coupled']==sb['reserved_coupled']==3932)
check('budget_and_no_pending',sb['reserved_coupled']<=7000 and sb['integration_intervals']<=17000000 and db['counts']=={'native':13500,'replay':13500} and db['pending'] is None and read(RUN/'pending.json')=={'phase':None,'unknown':False} and summary['new_native_and_replay_steps']==54500 and status['elapsed_s']<=10800)
check('new_execution_event_sequence',len(eventstages['public_proposals'])==5 and len(eventstages['native67_execution_and_independent_replay'])==1 and len(eventstages['actual_terminal_handoff_same_nine_obstacles'])==1 and eventstages['native67_execution_and_independent_replay'][0]['timestamp']<eventstages['actual_terminal_handoff_same_nine_obstacles'][0]['timestamp']<eventstages['independent_replay'][0]['timestamp'])
manifest=read(RUN/'dynamic/visuals/manifest.json');check('visual_manifest_all_hashes',all(sha(RUN/'dynamic/visuals'/p)==v['sha256'] for p,v in manifest['files'].items()))
paper=ROOT/'paper/PAPER_CLAIM_AUDIT.json';paper_audit=read(paper) if paper.exists() else 'unavailable'
evidence=dict(review_role='fresh independent paper architect/reviewer',new_physics_steps=0,new_coupled_calls=0,new_geometry_certification=False,checks=checks,failed_checks=[k for k,v in checks.items() if not v],source_identity=dict(producer_head=run['producer_head'],archive_sha256=sha(RUN/'executed_source.zip'),source_count=208,input_count=93,archive_mismatches=source_mismatch,working_mismatches=working_mismatch,input_mismatches=inputs_mismatch),static_replay=static_replay,dynamic_replay=dynamic_replay,handoff=handoff,absolute_time=[sn['time'][0],sn['time'][-1],dn['time'][0],dn['time'][-1]],array_shapes={label:{k:list(v.shape) for k,v in a.items()} for label,a in [('static',sn),('dynamic',dn)]},model_hash=contract['scene_model_hash'],old_bound_static_model_hash=oldselection['entries'][0]['task']['contract']['robot_hash'],static_native_sha256=sha(sp/'execution/native_trace.npz'),dynamic_native_sha256=sha(RUN/'dynamic/execution/native_trace.npz'),motion=motion,integration_layout=external,static_hold_position_mm=np.max(spose[shold],axis=0)*1000,static_hold_rotation_deg=np.max(srot[shold],axis=0)*180/np.pi,dynamic_position_mm=dist(dpose*1000),dynamic_rotation_deg=dist(drot*180/np.pi),dynamic_hold_position_mm=np.max(dpose[dhold],axis=0)*1000,static_clearance_mm=float(np.min(sg['minima'])*1000),dynamic_clearance_mm=float(np.min(dg['minima'])*1000),static_geometry_states=len(sg['minima']),dynamic_geometry_states=len(dg['minima']),dynamic_momentum_peak=mp,dynamic_momentum_threshold=1e-5+.002*me,static_momentum_peak=smp,static_gates=sreport['gates'],dynamic_gates=dreport['gates'],proposals=rows,waypoints=waypoints,segment_durations_s=dur,obstacle_relevance=obstacle_relevance,budget=dict(static_coupled=calls['coupled'],static_integration_intervals=sb['integration_intervals'],static_physics=calls['physics_steps'],dynamic_counts=db['counts'],elapsed_s=status['elapsed_s'],output_bytes=sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file()),pending=None),execution_events=eventstages,paper_claim_audit=paper_audit,limitations=['Geometry distances and maximum geometry step are existing saved observations, not recertified.','Static momentum exchange threshold is checked against saved audit; mass matrices were not recomputed.','Same static native hash is not used as proof of either reuse or freshness. Freshness rests on frozen call path, source/model/task identity, events and call counters plus saved raw execution/replay.','No resume or old-run audit performed.'])
write('inputs.json',dict(generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='current run plus frozen dependency identities only; historical bound files are not re-audited',files=inventory))
write('compact_evidence.json',evidence)
print(json.dumps(dict(check_count=len(checks),failed_checks=evidence['failed_checks'],waypoints=waypoints,budget=evidence['budget'],motion=motion,model_hash=evidence['model_hash'],old_bound_static_model_hash=evidence['old_bound_static_model_hash']),indent=2,default=plain))

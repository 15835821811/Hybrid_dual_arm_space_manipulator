"""Fresh review: saved-array algebra and identity only. No project imports or simulator.
Writes exclusively beside this script. No optimization, training, replay or resume.
"""
from pathlib import Path
import hashlib, json, os, subprocess, zipfile, datetime, xml.etree.ElementTree as ET
from collections import Counter
import numpy as np
from scipy.interpolate import BSpline
from scipy.spatial.transform import Rotation

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
RUN=ROOT/'artifacts/v7_1/fullchain_acceptance_001'
inputs={}; checks=[]; static=[]; dynamic=[]; blocked=[]
def plain(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,Path):return str(x)
    raise TypeError(type(x).__name__)
def digest(x):return hashlib.sha256(json.dumps(x,default=plain,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sha(p):
    p=Path(p);h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def remember(p):
    p=Path(p).resolve();key=str(p)
    if key not in inputs:
        s=p.stat();inputs[key]=dict(path=key,sha256=sha(p),bytes=s.st_size,mtime_utc=datetime.datetime.fromtimestamp(s.st_mtime,datetime.timezone.utc).isoformat())
    return p
def read(p):return json.loads(remember(p).read_text(encoding='utf-8-sig'))
def arr(p):
    with np.load(remember(p),allow_pickle=False) as z:return {k:z[k] for k in z.files}
def check(n,ok,detail=None):checks.append(dict(name=n,passed=bool(ok),detail=detail))
def close(a,b,tol=2e-11):return np.allclose(a,b,rtol=1e-9,atol=tol)
def maximum(a):return float(np.max(np.abs(a),initial=0))
def stats(a):return dict(mean=np.mean(a,axis=0),p95=np.percentile(a,95,axis=0),maximum=np.max(a,axis=0),final=a[-1])
def write(name,data):(HERE/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,default=plain,allow_nan=False)+'\n',encoding='utf-8')

run=read(RUN/'run.json'); ident=run['identity'];selection=ident['selection'];protocol=ident['protocol']
status=read(RUN/'status.json'); budget=read(RUN/'budget.json'); launch=read(RUN/'launcher.json')
physical=read(RUN/'physical_contract.json'); acceptance=physical['acceptance']; ri=physical['identity']
check('producer_commit',run['producer_head']=='370a2041a4af0e4615fae259b2e863d4b676e490')
check('terminal_status',status['status']=='FINISHED_WITH_BLOCKED_ITEMS' and status['stage']=='complete')
check('no_unresolved_work',budget['inflight'] is None and not (RUN/'run.lock').exists())
check('source_count',len(ident['source_hashes'])==125)
check('binding_count',len(ident['binding']['files'])==589)
check('frozen_selection',selection==read(ROOT/'ff_prior_fullchain_acceptance/evidence/selection.json'))
check('frozen_binding',ident['binding']==read(ROOT/'ff_prior_fullchain_acceptance/evidence/binding.json'))
source_zip=zipfile.ZipFile(remember(RUN/'executed_source.zip'))
check('source_archive_names',set(source_zip.namelist())==set(ident['source_hashes']))
git_differences=[]
for p,h in ident['source_hashes'].items():
    check('source_archive/'+p,hashlib.sha256(source_zip.read(p)).hexdigest()==h)
    check('source_current/'+p,inputs[str(remember(ROOT/p))]['sha256']==h)
    result=subprocess.run(['git','show',run['producer_head']+':'+p],cwd=ROOT,capture_output=True)
    if result.returncode or hashlib.sha256(result.stdout).hexdigest()!=h:git_differences.append(p)
for p,h in ident['binding']['files'].items():check('bound/'+p,inputs[str(remember(ROOT/p))]['sha256']==h)
for label in ['old_static_sources','old_dynamic_sources']:
    for p,h in selection[label].items():check(label+'/'+p,inputs[str(remember(ROOT/p))]['sha256']==h)
for oldname in ['static_fullchain_002','temporal_learning_001']:
    oldroot=ROOT/'artifacts/v7_1'/oldname; old=read(oldroot/'run.json')['identity']['source_hashes']
    with zipfile.ZipFile(remember(oldroot/'executed_source.zip')) as z:
        check('old_archive_names/'+oldname,set(z.namelist())==set(old))
        for p,h in old.items():check('old_archive/'+oldname+'/'+p,hashlib.sha256(z.read(p)).hexdigest()==h)
    check('old_manifest_selection/'+oldname,old==selection['old_static_sources' if oldname.startswith('static') else 'old_dynamic_sources'])
source_zip.close()

names=ri['robot']['low_level_joint_names']; mapping=ri['joint_index_map']
qi=np.array([mapping[n]['qpos'] for n in names]);vi=np.array([mapping[n]['dof'] for n in names]);ai=np.array([mapping[n]['actuator'] for n in names])
pattern=np.array([[0,1],[1,0],[1,0],[0,1],[0,1],[1,0],[1,0],[0,1],[0,1],[1,0],[1,0],[0,1]])/6
B=np.zeros((67,17));B[:60,:10]=np.kron(np.eye(5),pattern);B[60:,10:]=np.eye(7);Bp=np.linalg.pinv(B)
vlim=np.r_[np.full(10,.8),np.full(7,1.2)];alim=np.r_[np.full(10,2.5),np.full(7,4.)];torque=np.r_[np.full(60,4.),np.full(7,80.)]
lower=np.array(ri['planner_lower']);upper=np.array(ri['planner_upper']);knots=np.r_[np.zeros(4),np.arange(1,21)/21,np.ones(4)]
urdf=ET.parse(remember(ROOT/'dual_arm_space_robot_2026/urdf/dual_arm_space_robot_2026.urdf')).getroot()
joints={j.attrib['name']:j for j in urdf.findall('joint')}
def native_limit(n,key,default):
    limit=joints[n].find('limit')
    return float(default if limit is None or joints[n].get('type')=='continuous' else limit.get(key,default))
native_lower=np.array([native_limit(n,'lower','-inf') for n in names]);native_upper=np.array([native_limit(n,'upper','inf') for n in names])
entries={x['sample_id']:x for x in selection['entries']}
check('units_frozen_30',len(selection['units'])==30 and len({s['id'] for s in selection['units']})==30)
check('scene_methods_seeds_denominator',Counter((s['scene'],s['method']) for s in selection['units'])==Counter({(s,m):3 for s in protocol['scenes'] for m in protocol['methods']}))
check('actions_frozen',Counter(s['action'] for s in selection['units'])=={'NEW_STATIC_EXECUTION':6,'EXACT_HISTORICAL_REUSE':12,'BLOCKED':12})
for spec in selection['units']:
    uid=spec['id'];d=RUN/uid;result=read(d/'result.json');check(uid+'/frozen_spec',read(d/'spec.json')==spec)
    if spec['action']=='BLOCKED':
        check(uid+'/blocked_explicit',result['level']=='BLOCKED' and result['new_physics_steps']==0 and result['new_execution'] is False)
        if spec['scene']=='five_obstacle_to_drifting':check(uid+'/combo_not_frozen',spec['task_hash'] is None and spec['task_status']=='DEPENDENCY_BLOCKED_NOT_FROZEN_FOR_EXECUTION')
        check(uid+'/no_execution_files',not (d/'execution').exists());blocked.append(dict(id=uid,scene=spec['scene'],reason=result['reason'],task_status=result['task_status']));continue
    if spec['action']=='EXACT_HISTORICAL_REUSE':
        origin=ROOT/spec['execution_source'];record=ROOT/spec['source'];report=read(origin/'result.json');task=read(record/'task.json');task0=read(origin/'task.json')
        check(uid+'/report_exact',result['report']==report)
        check(uid+'/task_identity',digest(task)==spec['task_hash'] and task==task0)
        check(uid+'/native_hash',inputs[str(remember(origin/'native_trace.npz'))]['sha256']==spec['native_trace_sha256']==report['native_trace_sha256'])
        check(uid+'/reused_not_new',result['new_execution'] is False and result['new_physics_steps']==0 and result['scope']=='EXACT_HISTORICAL_EXECUTION_REUSE')
        check(uid+'/fullstate_report',np.isfinite(report['replay_full_integration_error']) and 0<=report['replay_full_integration_error']<=1e-10)
        check(uid+'/old_all_gates',report['status']=='EXECUTION_VERIFIED' and all(report['gates'].values()))
        for name in ['task.json','initial.npz','prepared.npz','proposal.npz','sampling.json','prepare.json','native_trace.npz','replay_trace.npz']:
            p=origin/name;check(uid+'/bound_component/'+name,p.relative_to(ROOT).as_posix() in ident['binding']['files'])
        check(uid+'/bound_model',all(('artifacts/v7_1/temporal_learning_001/learning/'+spec['component_method']+'/'+f) in ident['binding']['files'] for f in ['model.pt','training.json']))
        check(uid+'/bound_codec','artifacts/v7_1/temporal_learning_001/codec.json' in ident['binding']['files'])
        ini=arr(record/'initial.npz')['values'];ini0=arr(origin/'initial.npz')['values'];check(uid+'/initial_identity',np.array_equal(ini,ini0) and ini.shape==(836,))
        # Schema inspection only; no old numerical physics audit or re-replay.
        with np.load(origin/'native_trace.npz',allow_pickle=False) as z:
            check(uid+'/complete_state_schema',z['integration'].shape==(13501,836) and z['ctrl'].shape==(13500,67))
        with np.load(origin/'replay_trace.npz',allow_pickle=False) as z:check(uid+'/saved_fullstate_replay_schema',z['integration'].shape==(13501,836))
        if not spec['original_new_execution']:
            reuse=read(record/'reuse.json');check(uid+'/shared_binding',reuse['source']==spec['execution_source'] and reuse['identical_control_hash']==spec['native_trace_sha256'] and reuse['not_independent_replication'])
        dynamic.append(dict(id=uid,scene=spec['scene'],method=spec['method'],source=spec['source'],execution_source=spec['execution_source'],shared=not spec['original_new_execution'],position_mm=report['position_mm'],rotation_deg=report['rotation_deg'],full_integration_report_error=report['replay_full_integration_error']));continue
    # Six new static trajectories: all raw states and all saved geometry scans.
    e=d/'execution';report=read(e/'report.json');task=read(e/'frozen_task.json');calls=read(d/'calls.json');proof=read(d/'initial_audit.json');meta=report['metadata']
    n=arr(e/'native_trace.npz');r=arr(e/'replay_trace.npz');g=arr(e/'geometry_scans.npz');scans=read(e/'geometry_scans.json');initial=arr(e/'initial_state.npz')['values'];im=read(e/'initial_state.json')
    v=read(ROOT/spec['reference']);chosen=read(ROOT/spec['source']/'execution_selection.json')
    check(uid+'/reference_prespecified',chosen['candidate_id']==spec['candidate'] and v['eligible'])
    check(uid+'/task_bound',task==entries[spec['scene']]['task'] and task['task_hash']==spec['task_hash']==digest({k:v for k,v in task.items() if k!='task_hash'}))
    check(uid+'/report_copy',result['report']==report)
    check(uid+'/initial_hash',digest(dict(values=initial,model_hash=im['model_hash'],state_mask=im['state_mask']))==im['state_id']==meta['initial_state_id']==task['contract']['initial_state_id'])
    check(uid+'/initial_full_state',initial.shape==(836,) and close(n['integration'][0],initial))
    check(uid+'/full_state_dimensions',n['integration'].shape==(13751,836) and n['qpos'].shape==(13751,81) and n['qvel'].shape==(13751,79))
    check(uid+'/ctrl_dimensions',n['ctrl'].shape==n['raw_torque'].shape==(13750,67))
    check(uid+'/ctrl_hash',digest(n['ctrl'])==report['ctrl_hash'])
    replayerrs={k:maximum(n[k]-r[k]) for k in ['qpos','qvel','act','time','qacc_warmstart','integration']}
    check(uid+'/replay_all_states',all(v<=1e-10 for v in replayerrs.values()) and replayerrs==report['replay']['maximum_errors'] and np.array_equal(n['ctrl'],r['ctrl']))
    times=n['time']-n['time'][0];complete=len(times)==13751 and np.allclose(np.diff(times),.002,atol=1e-12,rtol=0) and abs(times[-1]-27.5)<1e-9 and meta['status']=='FINISHED'
    check(uid+'/native_and_replay_step_accounting',calls['physics_steps']==len(n['ctrl'])+len(r['ctrl'])==27500 and meta['physics_steps']==meta['planned_steps']==13750)
    Z=np.asarray(v['Z']);q0=np.array(task['condition']['initial_features']);C=np.vstack([np.repeat(q0[None],3,axis=0),Z[:18],np.repeat(Z[-1:],3,axis=0)]);curve=BSpline(knots,C,3,extrapolate=False)
    def ref(t,j=0):
        out=curve(np.clip(t/27,0,1),nu=j)/27**j
        return np.where(((t<0)|(t>27))[...,None],0,out) if j else out
    check(uid+'/reference_hash',digest(dict(C=C,knots=knots,degree=3,T=27.0,hold=.5))==v['reference_hash']==meta['reference_hash'])
    for j,key in enumerate(['reference_q','reference_qd','reference_qdd']):check(uid+'/'+key,close(n[key],ref(np.arange(13750)*.002,j)))
    q=n['qpos'][:,qi]@Bp.T;vel=n['qvel'][:,vi]@Bp.T;acc=np.diff(vel,axis=0)/.002;lowv=n['qvel'][:,vi];lowa=np.diff(lowv,axis=0)/.002
    check(uid+'/17_67_mapping',close(q,n['q_actual_17']) and close(n['qpos'][:,qi]-q@B.T,n['manifold_residual']))
    goals=np.asarray(task['condition']['goals']);pos=np.linalg.norm(n['ee'][...,:3,3]-goals[:,:3,3],axis=-1);rel=np.swapaxes(goals[:,:3,:3],-1,-2)@n['ee'][...,:3,:3];rot=Rotation.from_matrix(rel.reshape(-1,3,3)).magnitude().reshape(pos.shape);hold=times>=27-1e-9
    hp=pos[hold].max(axis=0);hr=rot[hold].max(axis=0)
    for key,val in [('final_position_error_m',pos[-1]),('final_rotation_error_rad',rot[-1]),('hold_max_position_error_m',hp),('hold_max_rotation_error_rad',hr)]:check(uid+'/'+key,close(val,report[key]))
    minima=np.array([x['minimum_clearance_m'] for x in scans]);check(uid+'/geometry_minima_all',np.array_equal(minima,g['minima']) and len(minima)==27501 and close(min(minima),report['minimum_clearance_m']))
    idx=np.array([x['state'] for x in scans]);frac=np.array([x['fraction'] for x in scans]);native=frac==0;mid=frac==.5
    check(uid+'/geometry_native_coverage',np.array_equal(idx[native],np.arange(13751)) and np.array_equal(g['qpos'][native],n['qpos']))
    check(uid+'/geometry_midpoint_coverage',np.array_equal(idx[mid],np.arange(13750)) and native.sum()==13751 and mid.sum()==13750)
    # Scalar and translation midpoints can be checked without MuJoCo; quaternion interpolation remains source-inspected.
    scalar=np.r_[np.arange(7,74),np.arange(74,77)];check(uid+'/geometry_midpoint_scalar_state',close(g['qpos'][mid][:,scalar],.5*(n['qpos'][:-1,scalar]+n['qpos'][1:,scalar])))
    check(uid+'/native_clearance_alignment',np.array_equal(n['clearance_m'],minima[native]))
    track=maximum(q-ref(times));manifold=maximum(n['qpos'][:,qi]-q@B.T);mom=n['base_generalized_momentum'];peak=np.array([np.max(np.linalg.norm(mom[:,s],axis=1)) for s in [slice(0,3),slice(3,6)]])
    check(uid+'/momentum_peak_raw',close(peak,report['momentum_peak_linear_angular']))
    check(uid+'/momentum_threshold_formula',close(1e-5+.002*np.array(report['momentum_exchange_scale_linear_angular']),report['momentum_threshold_linear_angular']))
    check(uid+'/tracking_manifold',close(track,report['tracking_max_rad']) and close(manifold,report['manifold_max_rad']))
    derived=dict(finite=all(np.isfinite(x).all() for x in n.values()),identity=im['model_hash']==meta['scene_model_hash']==task['contract']['robot_hash'] and report['pair_policy_hash']==task['contract']['pair_policy_hash'],complete_native_time=complete,dual_goal_hold=complete and np.all(hp<=.005) and np.all(hr<=np.pi/180),whole_body_clearance=min(minima)>=0,joint_limits=np.all(q>=lower) and np.all(q<=upper) and np.all(ref(times)>=lower) and np.all(ref(times)<=upper) and np.all(n['qpos'][:,qi]>=native_lower) and np.all(n['qpos'][:,qi]<=native_upper),derivative_limits=np.all(abs(vel)<=vlim) and np.all(abs(acc)<=alim) and np.all(abs(lowv)<=abs(B)@vlim) and np.all(abs(lowa)<=abs(B)@alim) and np.all(abs(ref(times,1))<=vlim) and np.all(abs(ref(times,2))<=alim),raw_torque_limits=np.all(abs(n['raw_torque'])<=torque+1e-12) and np.array_equal(n['ctrl'][:,ai],np.clip(n['raw_torque'],-torque,torque)),tracking=track<=.02,manifold=manifold<=.02,momentum=np.all(peak<=report['momentum_threshold_linear_angular']),independent_replay=all(v<=1e-10 for v in replayerrs.values()),no_external_wrench=meta['external_wrench_applied'] is False,no_hidden_reset=meta['robot_state_writes_during_step_loop']==0 and complete,geometry_step=report['maximum_geometry_step_m']<=.002+1e-12)
    for key,val in derived.items():check(uid+'/gate/'+key,bool(val)==report['gates'][key] and val)
    check(uid+'/grading',all(derived.values()) and report['level']=='EXECUTION_VERIFIED' and result['new_planning'] is False and report['training_eligible'] is False)
    base_move=float(np.linalg.norm(n['base'][:,:3]-n['base'][0,:3],axis=1).max());check(uid+'/free_base_motion',base_move>0 and close(base_move,report['maximum_base_translation_m']))
    static.append(dict(id=uid,method=spec['method'],seed=spec['seed'],candidate=spec['candidate'],native_steps=len(n['ctrl']),replay_steps=len(r['ctrl']),native_states=len(times),integration_width=836,checked_geometry_states=len(minima),replay_max_errors=replayerrs,hold_position_mm=hp*1000,hold_rotation_deg=hr*180/np.pi,all_time_position_mm=stats(pos*1000),hold_position_stats_mm=stats(pos[hold]*1000),hold_rotation_stats_deg=stats(rot[hold]*180/np.pi),minimum_clearance_mm=min(minima)*1000,recorded_max_geometry_step_mm=report['maximum_geometry_step_m']*1000,max_actual_velocity_ratio=float(np.max(abs(vel)/vlim)),max_actual_acceleration_ratio=float(np.max(abs(acc)/alim)),max_low_acceleration_ratio=float(np.max(abs(lowa)/(abs(B)@alim))),max_torque_ratio=float(np.max(abs(n['raw_torque'])/torque)),tracking_max_rad=track,manifold_max_rad=manifold,base_translation_mm=base_move*1000,reconstructed_gates=derived))
    print('checked',uid,flush=True)

review=read(ROOT/'paper/review-traces/experiment-result-to-claim/20261022_temporal_learning_run01/verdict.json')
check('old_full_state_review',review['validation']['saved_evidence_failures']==0 and any('836' in x and '16' in x for x in review['what_results_support']))
remember(ROOT/'ff_prior_temporal_learning/docs/ANALYSIS.md');remember(ROOT/'DIFFUSION_GOAL_CURRENT.md');remember(ROOT/'ff_prior_fullchain/docs/ROADMAP.md')
for p in ['ff_prior_fullchain_acceptance/handoff.py','ff_prior_static_goal_seed/tracker.py','ff_prior_static_goal_seed/physical_audit.py']:remember(ROOT/p)
software=read(ROOT/'ff_prior_fullchain_acceptance/evidence/software_tests.json');check('software_tests_record',len(software['checks'])==37 and all(software['checks'].values()))
steps=sum(x['native_steps']+x['replay_steps'] for x in static)
check('native_budget_exact',steps==budget['actual_native_replay_steps']==budget['reserved_native_replay_steps']==protocol['new_native_replay_steps']==165000)
check('wall_budget',status['elapsed_s']<=21600 and budget['elapsed_s']<=21600)
bytes_now=sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file());check('output_budget',bytes_now<=20*1024**3)
check('protocol_no_new_learning_data_test',protocol['new_training_updates']==protocol['new_proposals']==protocol['TRAIN_added']==0 and protocol['TEST'] is False)
check('historical_distinct_8',len({x['execution_source'] for x in dynamic})==8 and sum(x['shared'] for x in dynamic)==4)
results=dict(status='PASS' if all(x['passed'] for x in checks) else 'FAIL',check_count=len(checks),failures=[x for x in checks if not x['passed']],source_files=125,bound_inputs=589,archive_is_reproduction_source=True,producer_git_differences=git_differences,counts=dict(logical_denominator=30,verified_logical=18,new_static_verified=6,historical_dynamic_logical=12,historical_dynamic_unique=8,historical_shared_logical=4,blocked=12,failed_new=0,covered_scenes=3,total_scenes=5,per_method=dict(verified_logical=9,denominator=15,blocked=6)),static=static,dynamic=dynamic,blocked=blocked,resources=dict(native_steps=steps//2,replay_steps=steps//2,total_steps=steps,worker_elapsed_s=status['elapsed_s'],budget_elapsed_s=budget['elapsed_s'],recorded_output_bytes=budget['output_bytes'],actual_output_bytes=bytes_now,threads_contract=2,threads_runtime_observed=False),paper_claim_audit='unavailable' if not (ROOT/'paper/PAPER_CLAIM_AUDIT.json').exists() else 'present_requires_review',review_actions=dict(new_physics_steps=0,geometry_queries=0,optimizer_steps=0,training_updates=0,resume_calls=0),limitations=['Geometry distances and maximum geometry extent are source-inspected and reconciled from saved full scans/report; no new collision queries or geometric engine execution.','Momentum exchange scale is bound report data; raw saved momentum peak and threshold formula are independently checked.','No new simulator replay; all six saved independent replay arrays compared in full.','Dynamic old physics is not re-audited; exact identity, saved full-state schema and existing independent full-state review are checked.','Two-thread configuration is source-bound; no live process performance counters are available after exit.'],checks=checks)
write('raw_checks.json',results);write('inputs.json',dict(generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='full trace: new six raw static trajectories; exact dynamic reuse identities only',files=list(inputs.values())))
print(json.dumps({k:results[k] for k in ['status','check_count','failures','counts','resources','producer_git_differences']},ensure_ascii=False,default=plain))

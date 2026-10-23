"""Independent algebra-only audit of all saved arrays in balanced_restore_001.
No project imports, MuJoCo, model compile, physics, geometry queries or solver calls.
Historical bound files are hashed only; historical raw arrays are never opened.
"""
from pathlib import Path
import json, hashlib, zipfile, datetime, xml.etree.ElementTree as ET
import numpy as np
from scipy.interpolate import BSpline
from scipy.spatial.transform import Rotation

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
RUN=ROOT/'artifacts/v7_1/balanced_restore_001'
UNIT=RUN/'units/u00'
OUT={}; INPUTS={}; CHECKS=[]
def plain(v):
    if isinstance(v,np.ndarray): return v.tolist()
    if isinstance(v,np.generic): return v.item()
    if isinstance(v,Path): return str(v)
    raise TypeError(type(v).__name__)
def digest(v):
    return hashlib.sha256(json.dumps(v,default=plain,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def fh(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''): h.update(b)
    return h.hexdigest()
def track(p,role='reviewed'):
    p=p.resolve(); k=str(p.relative_to(ROOT))
    if k not in INPUTS:
        s=p.stat();INPUTS[k]=dict(path=str(p),sha256=fh(p),bytes=s.st_size,mtime_utc=datetime.datetime.fromtimestamp(s.st_mtime,datetime.timezone.utc).isoformat(),role=role)
    return INPUTS[k]['sha256']
def read(p):
    track(p);return json.loads(p.read_text(encoding='utf-8-sig'))
def load(p):
    track(p)
    with np.load(p,allow_pickle=False) as z: a={k:z[k] for k in z.files}
    OUT.setdefault('array_inventory',{})[str(p.relative_to(ROOT))]={k:dict(shape=v.shape,dtype=str(v.dtype),finite=int(np.isfinite(v).sum()),elements=v.size) for k,v in a.items()}
    return a
def ck(name,ok,detail=None): CHECKS.append(dict(name=name,passed=bool(ok),detail=detail))
def close(name,a,b,tol=1e-11):
    a,b=np.asarray(a),np.asarray(b)
    if a.shape!=b.shape: ck(name,False,dict(shapes=[a.shape,b.shape]));return
    d=float(np.max(np.abs(a-b),initial=0));ck(name,np.isfinite(d) and d<=tol,dict(max_abs_error=d,atol=tol))
def pose(a,g):
    a,g=np.asarray(a),np.asarray(g)
    p=np.linalg.norm(a[...,:3,3]-g[...,:3,3],axis=-1)
    r=Rotation.from_matrix((np.swapaxes(g[...,:3,:3],-1,-2)@a[...,:3,:3]).reshape(-1,3,3)).magnitude().reshape(p.shape)
    return p,r
def goalvec(a,g):
    a,g=np.asarray(a),np.asarray(g)
    shape=a.shape[:-3]
    p=((a[...,:3,3]-g[...,:3,3])/.005).reshape(*shape,6)
    r=Rotation.from_matrix((np.swapaxes(g[...,:3,:3],-1,-2)@a[...,:3,:3]).reshape(-1,3,3)).as_rotvec().reshape(*shape,6)/np.deg2rad(1)
    return np.concatenate([p,r],axis=-1)
def merit(g,d):
    v=np.minimum((np.asarray(d)-.0002)/.003,0)
    return .05*float(g@g)+(float(np.mean(v*v)+.25*np.max(v*v)) if len(v) else 0.)
def select(near):
    ranked=sorted(range(len(near)),key=lambda i:(near[i,3],*near[i,:3]));chosen=[];reason=[];pairs=set();bins=set()
    def take(i,r):
        if i not in chosen and len(chosen)<240: chosen.append(i);reason.append(r)
    for i in ranked:
        p=tuple(near[i,1:3].astype(int))
        if p not in pairs: take(i,'worst_for_geometry_pair');pairs.add(p)
    for i in ranked:
        b=min(63,int(near[i,0]*64))
        if b not in bins: take(i,'worst_for_phase_bin');bins.add(b)
    for i in ranked: take(i,'remaining_by_distance')
    order=np.argsort(chosen)
    return np.asarray(chosen)[order],[reason[i] for i in order]

run=read(RUN/'run.json'); ident=run['identity']; selection=read(ROOT/'ff_prior_balanced_restore/evidence/selection.json')
binding=read(ROOT/'ff_prior_balanced_restore/evidence/binding.json'); task=selection['entries'][0]['task']; goals=np.asarray(task['condition']['goals'])
for p in [ROOT/'ff_prior_balanced_restore/docs/EXPERIMENT_PLAN.md',ROOT/'ff_prior_balanced_tracking/docs/EXPERIMENT_PLAN.md']: track(p)
ck('producer_head',run['producer_head']=='5a21e08868cbff6a90d93e8225a43b77e8302bcd')
ck('selection_exact',selection==ident['selection']);ck('binding_exact',binding==ident['binding'])
source_mismatch=[]; bound_mismatch=[]
with zipfile.ZipFile(RUN/'executed_source.zip') as archive:
    ck('archive_file_set',set(archive.namelist())==set(ident['source_hashes']))
    for name,h in ident['source_hashes'].items():
        ah=hashlib.sha256(archive.read(name)).hexdigest();wh=track(ROOT/name,'frozen_source')
        if ah!=h or wh!=h: source_mismatch.append(dict(path=name,bound=h,archive=ah,current=wh))
for name,h in binding['files'].items():
    actual=track(ROOT/name,'bound_identity_only_no_old_raw_analysis')
    if h!=actual:bound_mismatch.append(dict(path=name,bound=h,current=actual))
ck('all_source_hashes',not source_mismatch,source_mismatch);ck('all_bound_input_hashes',not bound_mismatch,bound_mismatch)
track(RUN/'executed_source.zip')
OUT['provenance']=dict(source_files=len(ident['source_hashes']),bound_files=len(binding['files']),producer_head=run['producer_head'])
software=read(ROOT/'ff_prior_balanced_restore/evidence/software_tests.json');ck('software_test_bound_source',software['source_hashes']==ident['source_hashes']);ck('software_status',software['status']=='PASS')
for p in (ROOT/'ff_prior_balanced_restore/evidence').glob('*'):
    if p.is_file():track(p)
report=read(UNIT/'execution/report.json');contract=read(RUN/'physical_contract.json'); profile=contract['acceptance'];initial_audit=read(RUN/'scenes/balanced_2R5C/initial_audit.json')
validation=read(UNIT/'validated/validation.json');restore_result=read(UNIT/'restore/result.json')
ck('task_frozen_identity',digest({k:v for k,v in task.items() if k!='task_hash'})==task['task_hash'])
ck('execution_frozen_task',read(UNIT/'execution/frozen_task.json')==task)
ck('scene_transfer_record',all(initial_audit['scene_transfer']['checks'].values()) and initial_audit['scene_transfer']['added_world_geoms']==7)
ck('same_acceptance',profile==task['contract']['acceptance'])
names=contract['identity']['robot']['low_level_joint_names']; mapping=contract['identity']['joint_index_map']; qids=np.array([mapping[n]['qpos'] for n in names]);vids=np.array([mapping[n]['dof'] for n in names]);aids=np.array([mapping[n]['actuator'] for n in names])
B=np.zeros((67,17));B[:60,:10]=np.kron(np.eye(5),np.array([[0,1],[1,0],[1,0],[0,1],[0,1],[1,0],[1,0],[0,1],[0,1],[1,0],[1,0],[0,1]])/6);B[60:,10:]=np.eye(7);BP=np.linalg.pinv(B)
lower=np.asarray(contract['identity']['planner_lower']);upper=np.asarray(contract['identity']['planner_upper']);vlim=np.r_[np.full(10,.8),np.full(7,1.2)];alim=np.r_[np.full(10,2.5),np.full(7,4.)];tlim=np.r_[np.full(60,4.),np.full(7,80.)]
# URDF range parsing is source reading only, no model construction.
urdf=ROOT/'dual_arm_space_robot_2026/urdf/dual_arm_space_robot_2026.urdf';track(urdf,'model_source_range_readonly')
joints={j.attrib['name']:j for j in ET.parse(urdf).getroot().findall('joint')}
rng=np.array([[float(joints[n].find('limit').get('lower','-inf')),float(joints[n].find('limit').get('upper','inf'))] if joints[n].find('limit') is not None else [-np.inf,np.inf] for n in names]);limited=np.array([joints[n].attrib['type']!='continuous' and joints[n].find('limit') is not None for n in names])
q0=np.asarray(task['condition']['initial_features']);knots=np.r_[np.zeros(4),np.arange(1,21)/21,np.ones(4)]
def spline(z):return BSpline(knots,np.vstack([np.repeat(q0[None],3,0),z[:18],np.repeat(z[-1:],3,0)]),3,extrapolate=False)
def refeval(c,t,d=0):
    t=np.asarray(t);v=c.derivative(d)(np.clip(t/27,0,1))/27**d
    return np.where(((t<0)|(t>27))[...,None],0,v) if d else v
def motion(z,label):
    c=spline(z);bounds=[]
    for order,lim in [(0,None),(1,vlim),(2,alim)]:
        d=c.derivative(order);ctrl=d.c[:len(d.t)-d.k-1]/27**order
        ck(label+f'_polygon_{order}',(np.all(ctrl>=lower) and np.all(ctrl<=upper)) if order==0 else np.all(abs(ctrl)<=lim+1e-12))
        bounds.append(np.max(abs(ctrl),0))
    low=c.c@B.T;ck(label+'_native_ranges',np.all(low[:,limited]>=rng[limited,0]) and np.all(low[:,limited]<=rng[limited,1]))
    ck(label+'_boundary',max(np.max(abs(refeval(c,[0,27],d))) for d in [1,2])<1e-10)
    return bounds
guess=load(ROOT/'ff_prior_balanced_restore/evidence/guess.npz')['Z'];motion(guess,'guess');finalz=np.asarray(validation['Z']);motion(finalz,'validated');curve=spline(finalz)
reference_hash=digest(dict(C=curve.c,knots=knots,degree=3,T=27.,hold=.5))
ck('reference_hash_chain',reference_hash==validation['reference_hash']==restore_result['reference_hash']==report['metadata']['reference_hash'])
close('restore_finalZ_equals_validated',restore_result['Z'],finalz,0)
n=load(UNIT/'execution/native_trace.npz');rp=load(UNIT/'execution/replay_trace.npz');initial=load(UNIT/'execution/initial_state.npz')['values'];bound_initial=load(ROOT/'ff_prior_balanced_restore/evidence/initial_state.npz')['values'];ij=read(UNIT/'execution/initial_state.json')
ck('native_full836_shape',n['integration'].shape==(13751,836));ck('replay_full836_shape',rp['integration'].shape==(13751,836));ck('finite_all_native',all(np.isfinite(v).all() for v in n.values()));ck('finite_all_replay',all(np.isfinite(v).all() for v in rp.values()))
for k in rp:close('full_replay_'+k,n[k],rp[k],0)
close('bound_complete_initial836',initial,bound_initial,0);close('native_initial836',n['integration'][0],initial,0)
ck('initial_state_hash',digest(dict(values=initial,model_hash=ij['model_hash'],state_mask=ij['state_mask']))==ij['state_id']==task['contract']['initial_state_id'])
ck('task_scene_hash',ij['model_hash']==task['contract']['robot_hash']==report['scene_model_hash']==initial_audit['scene_transfer']['scene_robot_hash'])
ck('policy_hash',task['contract']['pair_policy_hash']==report['pair_policy_hash']==initial_audit['initial_geometry']['pair_policy_hash'])
dt=.002;t=n['time'];close('time_grid',t,t[0]+np.arange(13751)*dt,1e-9)
ck('full_time',abs(t[-1]-t[0]-27.5)<1e-9);close('state_time',n['integration'][:,0],t,0);close('state_qpos',n['integration'][:,1:82],n['qpos'],0);close('state_qvel',n['integration'][:,82:161],n['qvel'],0);close('state_warmstart',n['integration'][:,161:240],n['qacc_warmstart'],0)
close('state_ctrl_alignment',n['integration'][1:,240:307],n['ctrl'],0);ck('all_saved_external_forces_zero',np.count_nonzero(n['integration'][:,307:])==0)
q=n['qpos'][:,qids]@BP.T;v=n['qvel'][:,vids]@BP.T;a=np.diff(v,axis=0)/dt;lv=n['qvel'][:,vids];la=np.diff(lv,axis=0)/dt;low=n['qpos'][:,qids]
close('planner_q_decode',q,n['q_actual_17'],1e-15);close('manifold_array',low-q@B.T,n['manifold_residual'],1e-15)
for d,k in enumerate(['reference_q','reference_qd','reference_qdd']):close('reference_'+k,refeval(curve,np.arange(13750)*dt,d),n[k],1e-14)
ref=refeval(curve,t-t[0]);ck('all_joint_limits',np.all(q>=lower) and np.all(q<=upper) and np.all(ref>=lower) and np.all(ref<=upper) and np.all(low[:,limited]>=rng[limited,0]) and np.all(low[:,limited]<=rng[limited,1]))
ck('all_derivative_limits',np.all(abs(v)<=vlim) and np.all(abs(a)<=alim) and np.all(abs(lv)<=abs(B)@vlim) and np.all(abs(la)<=abs(B)@alim) and np.all(abs(refeval(curve,t-t[0],1))<=vlim) and np.all(abs(refeval(curve,t-t[0],2))<=alim))
ck('all_raw_torque_limits',np.all(abs(n['raw_torque'])<=tlim+1e-12));close('actual_ctrl_clipped_raw',n['ctrl'][:,aids],np.clip(n['raw_torque'],-tlim,tlim),0);ck('ctrl_hash',digest(n['ctrl'])==report['ctrl_hash'])
p,r=pose(n['ee'],goals);hold=t-t[0]>=27-1e-9
for key,val in [('final_position_error_m',p[-1]),('final_rotation_error_rad',r[-1]),('hold_max_position_error_m',p[hold].max(0)),('hold_max_rotation_error_rad',r[hold].max(0)),('tracking_max_rad',np.max(abs(q-ref))),('manifold_max_rad',np.max(abs(low-q@B.T))),('zero_momentum_max_residual',np.max(abs(n['base_generalized_momentum']))),('maximum_base_translation_m',np.max(np.linalg.norm(n['base'][:,:3]-n['base'][0,:3],axis=1)))]:close('report_'+key,val,report[key],1e-11)
peak=np.array([np.linalg.norm(n['base_generalized_momentum'][:,s],axis=1).max() for s in [slice(0,3),slice(3,6)]]);close('momentum_peaks',peak,report['momentum_peak_linear_angular']);threshold=1e-5+.002*np.asarray(report['momentum_exchange_scale_linear_angular']);close('momentum_threshold_formula',threshold,report['momentum_threshold_linear_angular']);ck('momentum_against_recorded_exchange_scale',np.all(peak<=threshold))
ck('dual_goal_hold_recomputed',np.all(p[hold]<=.005) and np.all(r[hold]<=np.deg2rad(1)))
OUT['execution']=dict(native_states=len(t),native_steps=len(n['ctrl']),state_elements=n['integration'].size,replay_max_abs=float(np.max(abs(n['integration']-rp['integration']))),initial_time=float(t[0]),terminal_time=float(t[-1]),hold_samples=int(hold.sum()),final_position_error_m=p[-1],hold_max_position_error_m=p[hold].max(0),hold_max_rotation_error_rad=r[hold].max(0),path_max_position_error_m=p.max(0),path_max_rotation_error_rad=r.max(0),planner_max_velocity=np.max(abs(v),0),planner_max_acceleration=np.max(abs(a),0),raw_torque_max=np.max(abs(n['raw_torque']),0),momentum_peaks=peak,initial836_digest=digest(initial),terminal836_digest=digest(n['integration'][-1]),satellite_qpos_constant=bool(np.array_equal(n['qpos'][:,74:],np.broadcast_to(n['qpos'][0,74:],n['qpos'][:,74:].shape))),satellite_velocity_max=float(np.max(abs(n['qvel'][:,73:]))))
# Full saved execution scan table: match every native state and midpoint, no new distances.
sc=load(UNIT/'execution/geometry_scans.npz');sr=read(UNIT/'execution/geometry_scans.json');nativeids=np.r_[0,np.arange(1,27501,2)];midids=np.arange(2,27501,2)
ck('full_scan_count',len(sr)==27501==len(sc['qpos']));close('all_scan_native_qpos',sc['qpos'][nativeids],n['qpos'],0);close('all_scan_native_minima',sc['minima'][nativeids],n['clearance_m'],0);close('all_scan_json_minima',sc['minima'],[x['minimum_clearance_m'] for x in sr],0)
mid=(n['qpos'][:-1]+n['qpos'][1:])/2
for sl in [slice(3,7),slice(77,81)]:
    x=n['qpos'][:-1,sl];y=n['qpos'][1:,sl]; y=np.where((x*y).sum(1)[:,None]<0,-y,y);m=x+y;mid[:,sl]=m/np.linalg.norm(m,axis=1)[:,None]
close('all_saved_geodesic_midpoint_qpos',sc['qpos'][midids],mid,1e-14)
ck('all_scan_state_fraction',all(sr[int(j)]['state']==i and sr[int(j)]['fraction']==0 for i,j in enumerate(nativeids)) and all(sr[int(j)]['state']==i and sr[int(j)]['fraction']==.5 for i,j in enumerate(midids)))
mini=int(sc['minima'].argmin());close('clearance_report',sc['minima'][mini],report['minimum_clearance_m'],0);ck('minimum_pair',sr[mini]['pair']==report['minimum_pair']);ck('all_scans_nonpenetrating',np.all(sc['minima']>=0));OUT['execution']['geometry']=dict(scans=len(sr),minimum_m=float(sc['minima'][mini]),minimum_record=sr[mini],minimum_scan_index=mini,native_minimum_m=float(n['clearance_m'].min()))

# All MODEL_VALID grids, including rejected geometry-refinement levels.
models={};modelrows=[]
for grid in [256,512,1024,2048,4096]:
    g=load(UNIT/f'validated/grid_{grid}.npz');models[grid]=g;num=read(UNIT/f'validated/grid_{grid}_numerical.json')
    ck(f'model_{grid}_finite',all(np.isfinite(v).all() for v in g.values()));close(f'model_{grid}_phase',g['phase'],np.linspace(0,1,grid+1),0);close(f'model_{grid}_initial_qpos',g['qpos'][0],n['qpos'][0],0)
    close(f'model_{grid}_planned_qpos',g['qpos'][:,qids],refeval(curve,g['phase']*27)@B.T,1e-14)
    ck(f'model_{grid}_numerical_record',num['solve_residual']<1e-10 and num['robot_momentum_residual']<1e-10 and num['quaternion_norm_error']<1e-12)
    if grid>256:
        s=load(UNIT/f'validated/grid_{grid}_geometry_scans.npz');j=read(UNIT/f'validated/grid_{grid}_geometry_scans.json');order=np.r_[np.arange(0,grid+1,2),np.arange(1,grid,2)]
        close(f'model_{grid}_all_scan_qpos',s['qpos'],g['qpos'][order],0);close(f'model_{grid}_all_scan_minima',s['minima'],[x['minimum_clearance_m'] for x in j],0)
        row=validation['validation_grids'][[512,1024,2048,4096].index(grid)];gc=grid//2+1
        close(f'model_{grid}_minimum',s['minima'].min(),row['geometry']['minimum_clearance_m'],0);close(f'model_{grid}_coarse_minimum',s['minima'][:gc].min(),row['geometry']['coarse']['minimum_clearance_m'],0);close(f'model_{grid}_mid_minimum',s['minima'][gc:].min(),row['geometry']['midpoints']['minimum_clearance_m'],0)
        ck(f'model_{grid}_nonpenetration',np.all(s['minima']>=0));ck(f'model_{grid}_reported_refinement_rule',row['gates']['geometry_refinement']==(row['geometry']['refinement_difference_m']<=.001 and row['geometry']['maximum_geometry_step_m']<=.002))
        co=models[grid//2];cp,cr=pose(g['ee'][-1],co['ee'][-1]);comp=validation['validation']['numerical']['comparisons'][[512,1024,2048,4096].index(grid)]
        close(f'model_{grid}_comparison_position',cp,comp['terminal_position_m']);close(f'model_{grid}_comparison_rotation',cr,comp['terminal_rotation_rad'])
        modelrows.append(dict(grid=grid,states=grid+1,minimum_m=float(s['minima'].min()),reported_geometry_step_m=row['geometry']['maximum_geometry_step_m'],geometry_refinement=row['gates']['geometry_refinement'],max_saved_center_alignment_m=float(np.linalg.norm(g['geom'][::2]-co['geom'],axis=-1).max()),max_saved_body_alignment_m=float(np.linalg.norm(g['body'][::2]-co['body'],axis=-1).max())))
pred=load(UNIT/'validated/prediction.npz')
for k in pred:close('prediction_final_grid_'+k,pred[k],models[4096][k],0)
gp,gr=pose(pred['ee'][-1],goals);close('model_final_goal_position',gp,validation['validation']['frozen_goal_position_error_m']);close('model_final_goal_rotation',gr,validation['validation']['frozen_goal_rotation_error_rad']);ck('model_final_goal_gate',max(gp)<=.005 and max(gr)<=np.deg2rad(1));ck('all_final_model_gates',all(validation['validation']['gates'].values()))
OUT['models']=dict(grids=modelrows,final_position_error_m=gp,final_rotation_error_rad=gr,numerical_intermediate_residuals_not_saved=True)

# Every restoration evaluation, near record, selected/omitted identity, Jacobian entry and probe.
evals=[]
for label,path in [('initial',UNIT/'restore/initial'),('iter00',UNIT/'restore/iter/00/trial0'),('iter01',UNIT/'restore/iter/01/trial0')]:
    e=load(path.with_suffix('.npz'));j=read(path.with_suffix('.json'));evals.append(e)
    ck(label+'_all_finite',all(np.isfinite(v).all() for v in e.values()));close(label+'_goal',goalvec(e['ee'][-1],goals),e['goal'])
    expected,reasons=select(e['near']);close(label+'_all_selected',expected,e['selected'],0);ck(label+'_reasons',reasons==j['selected_reasons']);close(label+'_all_omitted',np.setdiff1d(np.arange(len(e['near'])),expected),e['omitted'],0)
    ck(label+'_all_near_in_buffer',np.all(e['near'][:,3]<.004));ck(label+'_all_near_phases_saved',np.isin(e['near'][:,0],e['phase']).all());ck(label+'_all_near_geometry_ids',np.all(e['near'][:,1:3]==e['near'][:,1:3].astype(int)) and np.all((e['near'][:,1:3]>=0)&(e['near'][:,1:3]<161)))
    ck(label+'_all_minimum_pairs_count',len(j['minimum_pairs'])==len(e['phase']));close(label+'_minimum',e['minima'].min(),j['minimum_clearance_m'],0);close(label+'_Q',np.linalg.norm(e['goal'].reshape(4,3),axis=1).max(),j['Q']);close(label+'_merit',merit(e['goal'],e['minima']),j['merit'])
    close(label+'_initial_qpos',e['qpos'][0],n['qpos'][0],0);motion(e['Z'],label)
    # Each near pair's distance must be >= the minimum of that complete scan.
    pi=np.searchsorted(e['phase'],e['near'][:,0]);ck(label+'_near_not_less_than_minimum',np.all(e['near'][:,3]>=e['minima'][pi]-1e-14))
    OUT.setdefault('restore_evaluations',[]).append(dict(label=label,states=len(e['phase']),near=len(e['near']),selected=len(expected),omitted=len(e['omitted']),omitted_minimum_m=float(e['near'][e['omitted'],3].min()),minimum_m=float(e['minima'].min()),Q=j['Q'],merit=j['merit'],grid_history=j['grid_history']))
close('guess_enters_restoration_unchanged',guess,evals[0]['Z'],0);close('final_trial_enters_validation',evals[-1]['Z'],finalz,0)
coupled=5+sum(len(x['grid_history']) for x in OUT['restore_evaluations']);intervals=sum([256,512,1024,2048,4096])+sum(sum(h['intervals'] for h in x['grid_history']) for x in OUT['restore_evaluations'])
for it in [0,1]:
    path=UNIT/f'restore/iter/{it:02d}';j=load(path/'jacobian.npz');rec=read(path/'record.json');probes=read(path/'probes.json');prev=evals[it];nxt=evals[it+1]
    close(f'iter{it}_Z',j['Z'],prev['Z'],0);close(f'iter{it}_nextZ',j['next_Z'],nxt['Z'],0);close(f'iter{it}_active',j['active'],prev['near'][prev['selected'],:3],0)
    close(f'iter{it}_full_jg',(j['perturb_goal']-j['goal']).T/j['steps'],j['jg'],0);close(f'iter{it}_full_jd',(j['perturb_distances']-j['distances']).T/j['steps'],j['jd'],0)
    close(f'iter{it}_center_goal',goalvec(j['ee'],goals),j['goal']);close(f'iter{it}_all_perturbed_goal',goalvec(j['perturb_ee'],goals),j['perturb_goal'])
    ck(f'iter{it}_full_jacobian_finite',all(np.isfinite(v).all() for v in j.values()));close(f'iter{it}_steps_abs',abs(j['steps']),np.full(323,.0001),0)
    close(f'iter{it}_returned_solver_step',np.asarray(rec['solver']['returned_u'])*rec['radius_rad'],j['step'],0)
    trial=rec['trials'][0];close(f'iter{it}_accepted_step',j['Z']+trial['scale']*j['step'].reshape(19,17),j['next_Z'],0)
    before=merit(prev['goal'],prev['minima']);after=merit(nxt['goal'],nxt['minima']);prediction=merit(j['goal']+j['jg']@(trial['scale']*j['step']),j['distances']+j['jd']@(trial['scale']*j['step']));pb=merit(j['goal'],j['distances']);ratio=(before-after)/(pb-prediction)
    close(f'iter{it}_actual_reduction',before-after,trial['actual_reduction']);close(f'iter{it}_predicted_reduction',pb-prediction,trial['predicted_reduction']);close(f'iter{it}_ratio',ratio,trial['ratio']);ck(f'iter{it}_acceptance_rule',rec['accepted'] and trial['accepted'] and before-after>1e-10 and ratio>=.1)
    ck(f'iter{it}_no_probe_threshold',probes['acceptance_threshold'] is None)
    discrepancies=[]
    for pi,probe in enumerate(probes['observations']):
        close(f'iter{it}_probe{pi}_goal',goalvec(np.asarray(probe['ee']),goals),probe['goal'])
        if probe['kind']=='coordinate_half':direction=np.eye(323)[probe['coordinate']]*probe['step']
        else:direction=np.asarray(probe['direction'])*probe['step']*probe['sign']
        denom=np.linalg.norm(direction)
        discrepancies.append(dict(kind=probe['kind'],goal_linearization_error_norm=float(np.linalg.norm(np.asarray(probe['goal'])-j['goal']-j['jg']@direction)),distance_linearization_error_max_m=float(np.max(abs(np.asarray(probe['distances'])-j['distances']-j['jd']@direction))),step_norm=float(denom)))
        ck(f'iter{it}_probe{pi}_finite',all(np.isfinite(np.asarray(probe[k])).all() for k in ['goal','distances','ee','active_qpos','terminal_qpos']))
    sparse=rec['sparse_agreement'];interval_count=len(j['phase'])-1;coupled+=len(sparse)+323+len(probes['observations']);intervals+=sum(s['intervals'] for s in sparse)+(323+len(probes['observations']))*interval_count
    cp,cr=pose(j['ee'],prev['ee'][-1]);dd=np.max(abs(j['distances']-prev['near'][prev['selected'],3]));close(f'iter{it}_sparse_position',cp,sparse[-1]['position_m']);close(f'iter{it}_sparse_rotation',cr,sparse[-1]['rotation_rad']);close(f'iter{it}_sparse_distance',dd,sparse[-1]['distance_m']);ck(f'iter{it}_sparse_gate',max(cp)<=.001 and max(cr)<=np.deg2rad(.2) and dd<=.001)
    OUT.setdefault('restoration_iterations',[]).append(dict(iteration=it,solver_success=rec['solver']['success'],solver_message=rec['solver']['message'],solver_iterations=rec['solver']['iterations'],returned_motion_valid=rec['solver']['returned_motion_valid'],ratio=ratio,radius=rec['radius_rad'],next_radius=rec['next_radius_rad'],probe_diagnostics=discrepancies))
calls=read(UNIT/'calls.json');budget=read(RUN/'budget.json');ck('coupled_reconstructed',coupled==calls['coupled']==budget['actual_coupled']);ck('intervals_reconstructed',intervals==budget['integration_intervals']);OUT['cost']=dict(reconstructed_coupled=coupled,reconstructed_model_intervals=intervals,calls=calls)
native_budget=read(UNIT/'execution/native_budget.json');ck('native_budget_counts',native_budget['completed_observed']==native_budget['attempted_observed']==native_budget['reserved_upper_bound']==dict(execution=13750,replay=13750));ck('native_budget_finished',native_budget['finished']==['execution','replay'] and native_budget['pending'] is None)
ck('all_report_gates_true',len(report['gates'])==15 and all(report['gates'].values()))
# Save every reviewed endpoint value separately for exact downstream provenance review.
np.savez_compressed(HERE/'audited_endpoints.npz',initial836=n['integration'][0],terminal836=n['integration'][-1],terminal_qpos=n['qpos'][-1],terminal_qvel=n['qvel'][-1],terminal_reference_q=refeval(curve,27.5),terminal_reference_qd=refeval(curve,27.5,1))
audit_path=ROOT/'paper/PAPER_CLAIM_AUDIT.json'
OUT['paper_claim_audit']=read(audit_path) if audit_path.exists() else 'unavailable'
OUT['independent_reconstruction_limits']=['No new physical integration, model propagation, geometry query, optimization, neural inference or training was performed.','Saved distance minima and all associated states/identities were independently reduced and cross-checked; pairwise distance geometry is not recomputed.','Native geometry centers/rotations and model geom_rbound are not in original native arrays: original maximum_geometry_step_m is recorded evidence, not independently reconstructed in this script.','All stored base generalized momentum components are reduced independently; native full mass matrices and momentum_exchange arrays were not saved, so momentum values from qpos/qvel and relative scale cannot be independently reconstructed without dynamics evaluation.','Model numerical residual scalars are saved and pass their gates; underlying solves/mass matrices were not saved for independent residual reconstruction.','Near records are exhaustively partitioned and selection rules independently reproduced; saved minima/near tables cannot prove absence of a producer omission from an unsaved exact all-pair distance list.','No physical causal advantage is inferred from selected constraints, saved near distances, or later body shape observations.']
OUT['checks']=CHECKS;OUT['summary']=dict(checks=len(CHECKS),passed=sum(c['passed'] for c in CHECKS),failed=[c for c in CHECKS if not c['passed']])
(HERE/'core_numerical_audit.json').write_text(json.dumps(OUT,default=plain,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
(HERE/'inputs_core.json').write_text(json.dumps(dict(generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=list(INPUTS.values())),ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(OUT['summary'],default=plain,ensure_ascii=False,indent=2));print(json.dumps(OUT['execution'],default=plain,ensure_ascii=False,indent=2));print(json.dumps(OUT['cost'],default=plain,ensure_ascii=False,indent=2))

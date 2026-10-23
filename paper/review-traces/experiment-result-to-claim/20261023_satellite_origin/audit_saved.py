"""One bounded, independent saved-evidence audit. Never advances dynamics."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import sys,time,json,hashlib,zipfile,datetime
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT));os.chdir(ROOT)
OUT=Path(__file__).resolve().parent
RUN=ROOT/'artifacts/v7_1/moving_satellite_001';EX=RUN/'execution'
START=time.monotonic();INPUTS={};R={}
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def bind(p):
    p=Path(p);p=p if p.is_absolute() else ROOT/p
    INPUTS[str(p.relative_to(ROOT))]={'sha256':sha(p),'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns}
    return p
def read(p):return json.loads(bind(p).read_text(encoding='utf-8-sig'))
def arrays(p):
    with np.load(bind(p),allow_pickle=False) as z:return {k:z[k] for k in z.files}
def scalar(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    raise TypeError(type(x).__name__)
def save(name,x):(OUT/name).write_text(json.dumps(x,default=scalar,indent=2,allow_nan=False),encoding='utf-8')
def maxabs(x):return float(np.max(np.abs(x))) if np.size(x) else 0.
def note(s):print(s,flush=True)
def check(k,v):
    R.setdefault('checks',{})[k]=bool(v)
def peak2(x):return np.stack([np.linalg.norm(x[:,:3],axis=1),np.linalg.norm(x[:,3:],axis=1)],axis=1)
note('Bind new run, frozen sources and bound inputs')
run=read(RUN/'run.json');c=read('ff_prior_satellite_origin/evidence/contract.json')
budget=read(RUN/'budget.json');summary=read(RUN/'summary.json');status=read(RUN/'status.json')
auth=read('ff_prior_satellite_origin/evidence/authorization.json');cost=read('ff_prior_satellite_origin/evidence/budget_derivation.json')
tests=read('ff_prior_satellite_origin/evidence/software_tests.json');binding=read('ff_prior_satellite_origin/evidence/binding.json')
bind('ff_prior_satellite_origin/docs/EXPERIMENT_PLAN.md')
initial=arrays('ff_prior_satellite_origin/evidence/terminal.npz');runinitial=arrays(RUN/'initial.npz')
report=read(EX/'result.json');hand=read(RUN/'handoff.json');preclose=read(c['source_closeout'])
source_hashes=run['identity']['source_hashes'];bad_source=[];bad_zip=[]
with zipfile.ZipFile(bind(RUN/'executed_source.zip')) as z:
    check('archive_member_names',set(z.namelist())==set(source_hashes))
    for name,expected in source_hashes.items():
        p=bind(name)
        if INPUTS[str(p.relative_to(ROOT))]['sha256']!=expected:bad_source.append(name)
        if hashlib.sha256(z.read(name)).hexdigest()!=expected:bad_zip.append(name)
bad_bound=[]
for name,expected in binding['files'].items():
    p=ROOT/name
    actual=INPUTS.get(str(p.relative_to(ROOT)),{}).get('sha256') or sha(p)
    if actual!=expected:bad_bound.append(name)
R['identity']={'producer_head':run['producer_head'],'sources':len(source_hashes),'bound_inputs':len(binding['files']),
    'source_mismatches':bad_source,'archive_mismatches':bad_zip,'bound_input_mismatches':bad_bound,
    'predecessor_access':'closeout metadata only; inherited files hashed for identity but no old raw arrays loaded'}
check('frozen_identity',not bad_source and not bad_zip and not bad_bound and tests['source_hashes']==source_hashes)
from ff_prior.storage import digest
tmp=dict(c);h=tmp.pop('contract_hash');check('contract_digest',digest(tmp)==h)
check('terminal_digest',digest(initial['integration'])==c['source_terminal_digest'])
note('Load actual, replay and full satellite arrays')
n=arrays(EX/'native_trace.npz');r=arrays(EX/'replay_trace.npz');l=arrays(EX/'satellite_ledger.npz')
lm=read(EX/'satellite_ledger_metadata.json');g=arrays(EX/'geometry.npz');gp=read(EX/'geometry_pairs.json')
b=arrays(EX/'body_obstacle_trace.npz');bm=read(EX/'body_obstacle_metadata.json')
cc=arrays(EX/'controller_clearance.npz');cm=read(EX/'controller_clearance_metadata.json')
filters=read(RUN/'control_filter.json');events=read(EX/'body_filter_events.json')
R['shapes']={group:{k:list(v.shape) for k,v in data.items()} for group,data in [('native',n),('replay',r),('ledger',l),('geometry',g),('body',b),('constraints',cc)]}
t=n['time'];local=t-t[0];N=len(t)-1;dt=.002;tid=lm['target_body_id'];s=l['state'];F=l['xfrc_applied'][:,tid,:3]
check('terminal_all836',np.array_equal(n['integration'][0],initial['integration']) and np.array_equal(runinitial['integration'],initial['integration']))
check('terminal_qpos_qvel_qref',all(np.array_equal(n[k][0],initial[k]) for k in ['qpos','qvel']) and np.array_equal(n['qref'][0],initial['reference_q']))
check('native_time',N==14500 and len(n['ctrl'])==14500 and maxabs(np.diff(t)-dt)<1e-10 and abs(t[0]-54.5)<1e-9 and abs(t[-1]-83.5)<1e-9)
R['replay']={k:maxabs(n[k]-r[k]) for k in ['qpos','qvel','integration','ctrl']}
R['replay']['state_values_compared']=int(n['integration'].size)
check('full836_replay',n['integration'].shape==(14501,836) and r['integration'].shape==n['integration'].shape and R['replay']['integration']==0)
check('replay_full_input_identity',np.array_equal(r['xfrc_applied'],l['xfrc_applied']) and np.array_equal(r['qfrc_applied'],l['qfrc_applied']))
pos=np.linalg.norm(n['ee'][:,:,:3,3]-n['desired'][:,:,:3,3],axis=2)
rr=n['desired'][:,:,:3,:3]@np.swapaxes(n['ee'][:,:,:3,:3],-1,-2)
rot=Rotation.from_matrix(rr.reshape(-1,3,3)).magnitude().reshape(len(t),2)
hold=local>=27.5-1e-9
def stats(x):return dict(mean=x.mean(0),p95=np.percentile(x,95,axis=0),maximum=x.max(0),final=x[-1])
R['tracking']={'position_mm':stats(pos*1000),'rotation_deg':stats(np.rad2deg(rot)),'hold_samples':int(hold.sum()),'hold_position_max_mm':pos[hold].max(0)*1000,'hold_rotation_max_deg':np.rad2deg(rot[hold]).max(0)}
R['gates']={
 'complete_native_time':R['checks']['native_time'], 'independent_replay':max(R['replay'][k] for k in ['qpos','qvel'])<=1e-10,
 'full_integration_replay':R['checks']['full836_replay'],'actual_terminal_handoff':R['checks']['terminal_all836'],
 'absolute_time_continuity':maxabs(np.diff(t)-dt)<1e-10,'whole_body_clearance':float(n['clearance'].min())>=0,
 'dynamic_position':bool(np.all(pos<=.005)),'dynamic_rotation':bool(np.all(rot<=np.pi/180)),
 'relative_hold':bool(N==14500 and hold.any() and np.all(pos[hold]<=.005) and np.all(rot[hold]<=np.pi/180)),
 'tracking':maxabs(n['q17']-n['qref'])<=.02,'no_hidden_reset':lm['full_state_restore_count']==1 and R['checks']['full836_replay'],
 'finite':all(np.isfinite(a).all() for a in n.values())}
mom=peak2(n['momentum']);exchange=peak2(n['exchange']);threshold=1e-5+.002*exchange.max(0)
R['momentum']={'definition':'robot generalized base conjugate momentum M(q)*qvel at base DOFs; angular component expressed in base free-joint coordinates, not world-origin total angular momentum',
    'initial_vector':n['momentum'][0],'final_vector':n['momentum'][-1],'initial_norm':mom[0],'final_norm':mom[-1],
    'peak_norm':mom.max(0),'peak_indices':mom.argmax(0),'exchange_peak':exchange.max(0),'threshold':threshold,
    'initial_over_threshold':mom[0]>threshold,'peak_over_threshold_ratio':mom.max(0)/threshold,
    'peak_change_from_initial':peak2(n['momentum']-n['momentum'][0]).max(0),
    'exceed_state_count':(mom>threshold).sum(0),'first_exceed_index':[int(np.flatnonzero(mom[:,j]>threshold[j])[0]) if np.any(mom[:,j]>threshold[j]) else None for j in range(2)]}
R['gates']['momentum']=bool(np.all(mom.max(0)<=threshold))
mid=(np.arange(N)+.5)*dt;expected=np.zeros((N,3));expected[:1000,2]=.02*(1-np.cos(np.pi*mid[:1000]))
imp=np.vstack([np.zeros(3),np.cumsum(F*dt,axis=0)])
ang=np.vstack([np.zeros(3),np.cumsum(np.cross(l['force_point_world'],F)*dt,axis=0)])
work=np.r_[0,np.cumsum(dt*np.sum(F*(s[:-1,12:15]+s[1:,12:15])*.5,axis=1))]
mass=lm['satellite_mass_kg'];clipped=np.minimum(local,2);vel=.02/mass*(clipped-np.sin(np.pi*clipped)/np.pi)
z=.02/mass*(clipped**2/2+(np.cos(np.pi*clipped)-1)/np.pi**2)+.04/mass*np.maximum(local-2,0)
ep=s[0,:3]+local[:,None]*s[0,12:15];ep[:,2]+=z;ev=np.tile(s[0,12:15],(len(t),1));ev[:,2]+=vel
other=l['xfrc_applied'].copy();other[:,tid]=0
R['satellite']={'mass_kg':mass,'force_count':int(np.count_nonzero(np.linalg.norm(F,axis=1))), 'force_max_N':maxabs(F),'force_formula_max_error':maxabs(F-expected),
 'impulse_formula_error':maxabs(imp-l['impulse']),'angular_impulse_formula_error':maxabs(ang-l['angular_impulse']),'work_formula_error_J':maxabs(work-l['work_J']),
 'force_point_vs_prestate_COM_error_m':maxabs(l['force_point_world']-s[:-1,:3]),'linear_balance_max':maxabs(s[:,18:21]-s[0,18:21]-imp),
 'angular_balance_max':maxabs(s[:,21:24]-s[0,21:24]-ang),'work_energy_balance_max_J':maxabs(s[:,24]-s[0,24]-work),
 'analytic_position_error_m':maxabs(s[:,:3]-ep),'analytic_velocity_error_m_s':maxabs(s[:,12:15]-ev),
 'coast_velocity_change':maxabs(s[1000:,12:18]-s[1000,12:18]),'displacement_m':s[-1,:3]-s[0,:3],
 'initial_velocity':s[0,12:15],'final_velocity':s[-1,12:15],'impulse_Ns':imp[-1],'angular_impulse_Nms':ang[-1],'work_J':work[-1],
 'linear_momentum_definition_error':maxabs(s[:,18:21]-mass*s[:,12:15]),'angular_velocity_max':maxabs(s[:,15:18])}
st=R['satellite'];R['gates'].update({
 'satellite_ledger_complete':bool(s.shape==(14501,25) and all(np.isfinite(x).all() for x in l.values())),
 'no_robot_external_wrench':not np.count_nonzero(other) and not np.count_nonzero(l['qfrc_applied']),
 'authorized_satellite_force':st['force_formula_max_error']<=1e-14 and not np.count_nonzero(l['xfrc_applied'][:,tid,3:]) and st['force_max_N']<=.04,
 'coast_no_external_wrench':not np.count_nonzero(l['xfrc_applied'][1000:]),
 'satellite_linear_impulse_balance':st['linear_balance_max']<=1e-9,'satellite_angular_impulse_balance':st['angular_balance_max']<=1e-9,
 'satellite_work_energy_balance':st['work_energy_balance_max_J']<=1e-9,'satellite_analytic_origin':st['analytic_position_error_m']<=3e-6 and st['analytic_velocity_error_m_s']<=1e-8,
 'satellite_inertial_coast':st['coast_velocity_change']<=1e-10,
 'actual_nonzero_satellite_motion':np.linalg.norm(s[-1,12:15]-s[0,12:15])>.0019 and np.linalg.norm(s[-1,:3]-s[0,:3])>.055,
 'full_origin_and_coast':N==14500 and abs(local[-1]-29)<1e-9})
note('Compile metadata; static kinematics and mass arithmetic for every saved native state')
import mujoco
def prohibited(*a,**k):raise RuntimeError('REVIEW_FORBIDS_DYNAMICS_OR_INTEGRATION')
for name in ['mj_step','mj_step1','mj_step2','mj_forward','mj_inverse','mj_Euler','mj_RungeKutta','mj_implicit','mj_integratePos']:
    if hasattr(mujoco,name):setattr(mujoco,name,prohibited)
from ff_prior.robot_adapter import RobotAdapter
from ff_prior.scenes import SceneSpec
robot=RobotAdapter(scene=SceneSpec.from_dict(c['scene']));model=robot.model;data=mujoco.MjData(model)
lowq=n['qpos'][:,robot.qpos_ids];lowv=n['qvel'][:,robot.dof_ids];v=lowv@robot.Bplus.T;a=np.diff(v,axis=0)/dt;la=np.diff(lowv,axis=0)/dt
limited=model.jnt_limited[robot.joints].astype(bool);limits=model.jnt_range[robot.joints][limited]
ratios={'actual_planner_velocity':maxabs(v/robot.spec.planner_velocity_limits),'actual_planner_acceleration':maxabs(a/robot.spec.planner_acceleration_limits),
 'actual_lowlevel_velocity':maxabs(lowv/(abs(robot.B)@robot.spec.planner_velocity_limits)), 'actual_lowlevel_acceleration':maxabs(la/(abs(robot.B)@robot.spec.planner_acceleration_limits)),
 'command_velocity':maxabs(n['reference_qd']/robot.spec.planner_velocity_limits),'command_acceleration':maxabs(n['reference_qdd']/robot.spec.planner_acceleration_limits),
 'raw_torque':maxabs(n['raw_torque']/robot.spec.torque_limits)}
R['motion']={'ratios':ratios,'manifold_max_rad':maxabs(lowq-n['q17']@robot.B.T),'tracking_max_rad':maxabs(n['q17']-n['qref']),
 'q17_reconstruction_error':maxabs(n['q17']-lowq@robot.Bplus.T),'reference_integration_error':maxabs(np.diff(n['qref'],axis=0)-dt*n['reference_qd']),
 'raw_control_error':maxabs(n['raw_torque']-n['ctrl'][:,robot.actuator_ids]),
 'command_acceleration_error':maxabs(np.diff(np.vstack([initial['reference_qd'],n['reference_qd']]),axis=0)/dt-n['reference_qdd'])}
R['gates']['joint_limits']=bool(np.all(n['q17']>=robot.lower) and np.all(n['q17']<=robot.upper) and np.all(n['qref']>=robot.lower) and np.all(n['qref']<=robot.upper) and np.all(lowq[:,limited]>=limits[:,0]) and np.all(lowq[:,limited]<=limits[:,1]))
R['gates']['derivative_limits']=all(ratios[k]<=1 for k in ratios if k!='raw_torque')
R['gates']['raw_torque']=bool(np.all(abs(n['raw_torque'])<=robot.spec.torque_limits+1e-12))
R['gates']['manifold']=R['motion']['manifold_max_rad']<=.02
M=np.zeros((model.nv,model.nv));pm=None;pr=None;geommax=0.;interp=0
mmerr=exerr=eeerr=bounderr=bodyerr=0.;worldangular=[];geompositions=[]
pair=np.array([[p['geom_a'],p['geom_b']] for p in cm['pairs']]);tick=0
for i in range(len(t)):
    if time.monotonic()-START>480:raise TimeoutError('BOUND_REVIEW_480_SECONDS')
    data.qpos[:]=n['qpos'][i];data.qvel[:]=n['qvel'][i]
    mujoco.mj_kinematics(model,data);mujoco.mj_comPos(model,data);mujoco.mj_crb(model,data);mujoco.mj_fullM(model,M,data.qM)
    mmerr=max(mmerr,maxabs((M@data.qvel)[robot.base_dofs]-n['momentum'][i]))
    exerr=max(exerr,maxabs(M[np.ix_(robot.base_dofs,robot.dof_ids)]@data.qvel[robot.dof_ids]-n['exchange'][i]))
    eeerr=max(eeerr,maxabs(robot.ee(data)-n['ee'][i]))
    gm=data.geom_xpos.copy();gr=data.geom_xmat.reshape(-1,3,3).copy()
    if pm is not None:
        angle=Rotation.from_matrix(gr@pr.transpose(0,2,1)).magnitude()
        motion=float(np.max(np.linalg.norm(gm-pm,axis=1)+angle*model.geom_rbound));parts=max(2,int(np.ceil(motion/.002)))
        geommax=max(geommax,motion/parts);interp+=parts-1
    pm,pr=gm,gr
    baseR=data.xmat[robot.base_body].reshape(3,3)
    worldangular.append(baseR@n['momentum'][i,3:]+np.cross(data.xpos[robot.base_body],n['momentum'][i,:3]))
    if i%10==0:
        bodyerr=max(bodyerr,maxabs(data.xpos-b['body_positions'][tick]));tick+=1
        if i<N:
            bounds=np.linalg.norm(gm[pair[:,0]]-gm[pair[:,1]],axis=1)-model.geom_rbound[pair[:,0]]-model.geom_rbound[pair[:,1]]
            bounderr=max(bounderr,maxabs(bounds-cc['sphere_lower_bound_m'][i//10]))
    if i and i%5000==0:note('Static reconstruction '+str(i))
R['static_reconstruction']={'momentum_max_error':mmerr,'exchange_max_error':exerr,'ee_pose_max_error':eeerr,'sphere_bound_max_error':bounderr,'body_positions_max_error':bodyerr,'geometry_step_m':geommax,'interpolation_states':interp,'native_states':len(t),'new_physics_steps':0,'static_kinematics_states':len(t)}
wa=np.array(worldangular);R['momentum']['world_origin_angular_initial']=wa[0];R['momentum']['world_origin_angular_final']=wa[-1];R['momentum']['world_origin_angular_change_peak']=float(np.linalg.norm(wa-wa[0],axis=1).max())
R['gates']['geometry_step']=geommax<=.002+1e-12
check('all_state_static_reconstruction',max(mmerr,exerr,eeerr,bounderr,bodyerr)<1e-10)
check('geometry_spacing_matches',abs(geommax-report['maximum_geometry_step_m'])<1e-12 and interp==report['interpolation_states'])
expected_min=np.minimum(g['minima'][1::2],g['minima'][2::2]) if len(g['minima'])==2*N+1 else None
R['geometry']={'scan_count':len(g['minima']),'minimum_m':float(g['minima'].min()),'native_minimum_m':float(n['clearance'].min()),'minimum_scan_index':int(g['minima'].argmin()),'pair_catalog_size':len(pair),'scan_minima_identity_error':maxabs(n['clearance'][1:]-expected_min) if expected_min is not None else None,'minimum_pair':gp[int(g['minima'].argmin())], 'exact_distance_requeries':0}
check('geometry_saved_state_alignment',len(g['qpos'])==2*N+1 and np.array_equal(g['qpos'][0],n['qpos'][0]) and np.array_equal(g['qpos'][1::2],n['qpos'][1:]))
check('geometry_saved_minima_alignment',expected_min is not None and np.array_equal(n['clearance'][1:],expected_min) and n['clearance'][0]==g['minima'][0])
note('Reconcile all pair dispositions, active constraints and body evidence')
dsp=cc['disposition'];bounds=cc['sphere_lower_bound_m'];dist=cc['distance_or_lower_bound_m'];exact=cc['exact_query_mask'];off=cc['row_offsets'];idx=cc['row_pair_index']
check('pair_selection',np.array_equal(exact,bounds<=.09) and np.array_equal(dsp==0,~exact) and np.array_equal(dsp==2,exact&(dist<=.08)) and not np.any(dsp==3))
check('omitted_lower_bound_identity',np.array_equal(dist[~exact],bounds[~exact]))
slacks=[];before=[];eventerr=0.;commanderr=0.;counts=[];eventtimeerr=0.;rowidentity=True
for j in range(1450):
    a0,a1=int(off[j]),int(off[j+1]);ix=idx[a0:a1];J=cc['jacobian17'][a0:a1];rhs=cc['lower_without_ramp_buffer'][a0:a1]+.008
    desired=np.array(filters[j]['desired']);filtered=np.array(filters[j]['filtered'])
    slack=J@filtered-rhs;prior=J@desired-rhs;slacks.extend(slack);before.extend(prior);counts.append(len(ix))
    rowidentity &= np.array_equal(ix,np.flatnonzero(dsp[j]==2))
    commanderr=max(commanderr,maxabs(filtered-n['reference_qd'][j*10+9]),maxabs(desired-b['command_before'][j]),maxabs(filtered-b['command_after'][j]))
    eventtimeerr=max(eventtimeerr,abs(events[j]['time']-t[j*10]),abs(filters[j]['time']-t[j*10]))
    erows=events[j]['active_constraints']
    rowidentity &= len(erows)==len(ix)
    for h,e in enumerate(erows):
        rowidentity &= int(e['pair_index'])==int(ix[h])
        eventerr=max(eventerr,maxabs(J[h]-np.array(e['jacobian17'])),abs(slack[h]-e['filtered_slack']),abs(prior[h]-e['unfiltered_slack']),abs(rhs[h]-e['lower_rhs']))
R['constraints']={'pair_time_entries':int(dsp.size),'pairs':len(pair),'ticks':len(dsp),'active_rows':len(idx),'disposition_counts':{str(k):int(np.count_nonzero(dsp==k)) for k in range(4)},
 'far_omitted_min_lower_bound_m':float(bounds[~exact].min()),'queried_cap_value_count':int(np.count_nonzero(exact & (dist>=.09))),
 'minimum_filtered_slack':float(np.min(slacks)),'minimum_unfiltered_slack':float(np.min(before)),
 'binding_rows':int(np.count_nonzero(np.abs(slacks)<=1e-6)),'violated_unfiltered_rows':int(np.count_nonzero(np.array(before)<-1e-8)),
 'geometry_projection_ticks':sum(bool(x.get('geometry_projection_required')) for x in filters),
 'slsqp_ticks':sum(bool(x.get('slsqp_used')) for x in filters),'event_arithmetic_max_error':eventerr,'command_identity_error':commanderr,'event_time_error':eventtimeerr}
check('all_active_row_event_identity',rowidentity and eventerr<1e-10 and commanderr<1e-12 and eventtimeerr<1e-9)
check('active_constraint_feasibility',np.min(slacks)>=-1e-6)
check('constraint_rhs_algebra',maxabs(cc['lower_without_ramp_buffer']+8*cc['margin_m']+cc['drift'])<1e-12)
check('body_tick_alignment',len(b['time'])==1451 and np.array_equal(b['native_index'],np.arange(0,N+1,10)) and np.array_equal(b['time'],t[::10]))
connector_distance=np.linalg.norm(b['connector'][...,1,:]-b['connector'][...,0,:],axis=-1)
R['body']={'samples':len(b['time']),'obstacles':len(bm['obstacles']),'links':len(bm['links']),'distance_entries':int(b['distance_m'].size),
 'connector_distance_error':maxabs(connector_distance-np.abs(b['distance_m'])),'queries_reported':bm['query_count'],'queries_expected':1451*7*sum(len(x['geom_ids']) for x in bm['links']),
 'per_obstacle_minimum':[],'terminal_command_nan':bool(np.isnan(b['command_before'][-1]).all() and np.isnan(b['command_after'][-1]).all()),'final_event_no_active_rows':len(events[-1]['active_constraints'])==0}
for oi,obs in enumerate(bm['obstacles']):
    sub=b['distance_m'][:,oi];ii,li=np.unravel_index(np.argmin(sub),sub.shape)
    R['body']['per_obstacle_minimum'].append({'obstacle':obs['object_id'],'distance_m':float(sub[ii,li]),'native_index':int(b['native_index'][ii]),'body':bm['links'][li]['body_name'],'arm':bm['links'][li]['arm_name'],'under_10mm_entries':int(np.count_nonzero(sub<.01))})
sk=b['body_positions'][:,bm['continuum_body_ids']];seg=np.diff(sk,axis=1);lens=np.linalg.norm(seg,axis=2)
turn=np.arctan2(np.linalg.norm(np.cross(seg[:,:-1],seg[:,1:]),axis=2),np.sum(seg[:,:-1]*seg[:,1:],axis=2));cur=turn/(.5*(lens[:,:-1]+lens[:,1:]))
R['body']['curvature_reconstruction_error']=maxabs(cur-b['continuum_curvature'])
check('body_evidence_arithmetic',R['body']['connector_distance_error']<1e-9 and R['body']['queries_reported']==R['body']['queries_expected'] and R['body']['curvature_reconstruction_error']<1e-10)
check('bound_reference_frames',maxabs(n['desired'][:,0]-np.array(c['hold_continuum_pose']))<1e-14 and maxabs(n['desired'][:,1,:3,3]-(s[:,:3]+np.einsum('nij,j->ni',s[:,3:12].reshape(-1,3,3),c['grasp_local'])))<1e-12)
R['gates']={k:bool(v) for k,v in R['gates'].items()}
R['gate_disagreements']={k:{'review':R['gates'].get(k),'producer':v} for k,v in report['gates'].items() if R['gates'].get(k)!=v}
R['failed_gates']=[k for k,v in R['gates'].items() if not v]
R['budget']={'native_steps':budget['counts']['native'],'replay_steps':budget['counts']['replay'],'total_native':sum(budget['counts'].values()),'inflight':budget['inflight'],'pending':budget['pending'],'coupled':budget['extra_coupled'],'review_physics_steps':0,'review_replay_steps':0,'remaining_native_after_actual':cost['original_remaining']['native_including_replay']-sum(budget['counts'].values())}
check('physical_budget',R['budget']['total_native']==29000 and budget['inflight'] is None and budget['pending'] is None)
for x in read(EX/'transitions.json'):
    q=arrays(EX/f"transition_{int(x['time']*10)}.npz");i=x['index'];check('transition_'+str(i),all(np.array_equal(q[k],n[k][i-1:i+2]) for k in q))
R['paper_claim_audit']='unavailable' if not (ROOT/'paper/PAPER_CLAIM_AUDIT.json').exists() else read('paper/PAPER_CLAIM_AUDIT.json')
R['raw_post_hash_mismatches']=[name for name,meta in INPUTS.items() if (name.startswith('artifacts\\v7_1\\moving_satellite_001') or name.startswith('ff_prior_satellite_origin\\evidence')) and sha(ROOT/name)!=meta['sha256']]
R['elapsed_s']=time.monotonic()-START
R['all_checks_pass']=all(R['checks'].values())
save('audit_results.json',R);save('inputs.json',{'generated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'new run only; no old raw trajectory reanalysis','files':INPUTS,'bound_manifest':binding,'run_source_hashes':source_hashes})
note(json.dumps({'elapsed_s':R['elapsed_s'],'all_checks_pass':R['all_checks_pass'],'failed_checks':[k for k,v in R['checks'].items() if not v],'failed_gates':R['failed_gates'],'gate_disagreements':R['gate_disagreements'],'momentum':R['momentum'],'satellite':R['satellite'],'tracking':R['tracking'],'static_reconstruction':R['static_reconstruction'],'constraints':R['constraints'],'body':R['body'],'geometry':R['geometry']},default=scalar))

"""Independent saved-evidence audit. No MuJoCo import, physics, FK, optimization or replay.
Only this run's arrays are interpreted; inherited inputs are hashed, with narrow metadata
and the newly frozen prefix read for source identity. All outputs stay in this trace.
"""
from pathlib import Path
import os, json, hashlib, zipfile, subprocess, sys, re, csv
from datetime import datetime, timezone
import xml.etree.ElementTree as ET
import numpy as np
from scipy.interpolate import BSpline
from scipy.spatial.transform import Rotation

TRACE=Path(__file__).resolve().parent
ROOT=TRACE.parents[3]
RUN=ROOT/'artifacts/v7_1/horizon_restore_001'
UNIT=RUN/'units/u00'
checks=[]; accessed={}; inventory=[]; notes=[]; domains={}; derivatives=[]
def plain(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,Path):return str(v)
    raise TypeError(type(v).__name__)
def digest(v):return hashlib.sha256(json.dumps(v,default=plain,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def touch(p,role='read'):
    p=Path(p).resolve();accessed.setdefault(str(p),set()).add(role);return p
def js(p):return json.loads(touch(p).read_text(encoding='utf-8-sig'))
def raw(p):return touch(p).read_bytes()
def npz(p):
    with np.load(touch(p),allow_pickle=False) as f:a={k:f[k] for k in f.files}
    inventory.append({'path':str(Path(p).relative_to(ROOT)),'arrays':{k:{'shape':v.shape,'elements':v.size,'dtype':str(v.dtype)} for k,v in a.items()}})
    check(str(Path(p).relative_to(ROOT))+'/finite',all(np.isfinite(v).all() for v in a.values()))
    return a
def check(name,ok,**detail):checks.append(dict(name=name,passed=bool(ok),**detail))
def close(name,a,b,atol=1e-11,rtol=1e-10):
    a=np.asarray(a);b=np.asarray(b)
    same=a.shape==b.shape
    delta=float(np.max(abs(a-b),initial=0)) if same else None
    check(name,same and np.allclose(a,b,atol=atol,rtol=rtol),shape_a=a.shape,shape_b=b.shape,max_abs_difference=delta,atol=atol,rtol=rtol)
def eq(name,a,b):check(name,a==b)
def write(p,data):p.write_text(json.dumps(data,default=plain,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

info=js(RUN/'run.json');ident=info['identity'];P=ident['protocol'];sel=ident['selection'];task=sel['entries'][0]['task']
q0=np.asarray(task['condition']['initial_features']);goals=np.asarray(task['condition']['goals']);profile=task['contract']['acceptance']
physical=js(RUN/'physical_contract.json');report=js(UNIT/'execution/report.json');validation=js(UNIT/'validated/validation.json')
fit=npz(UNIT/'fit.npz');fitmeta=js(UNIT/'fit.json');source=ident['source_hashes'];binding=ident['binding']['files']
print('identity',len(source),len(binding),flush=True)
with zipfile.ZipFile(touch(RUN/'executed_source.zip')) as z:
    eq('source/archive_file_set',set(z.namelist()),set(source))
    for p,h in source.items():
        eq('source/zip/'+p,hashlib.sha256(z.read(p)).hexdigest(),h)
        eq('source/current/'+p,sha(touch(ROOT/p,'source_hash')),h)
        blob=subprocess.run(['git','show',info['producer_head']+':'+p],cwd=ROOT,capture_output=True,check=True).stdout
        eq('source/production_commit_normalized/'+p,blob.replace(b'\r\n',b'\n'),z.read(p).replace(b'\r\n',b'\n'))
for p,h in binding.items():eq('binding/'+p,sha(touch(ROOT/p,'hash_only_inherited_input')),h)
eq('identity/selection',js(ROOT/'ff_prior_horizon_restore/evidence/selection.json'),sel)
eq('identity/binding',js(ROOT/'ff_prior_horizon_restore/evidence/binding.json'),ident['binding'])
tests=js(ROOT/'ff_prior_horizon_restore/evidence/software_tests.json')
eq('identity/software_tests_source_binding',tests['source_hashes'],source)
eq('identity/frozen_task_digest',digest({k:v for k,v in task.items() if k!='task_hash'}),P['task_hash'])
eq('identity/execution_frozen_task',js(UNIT/'execution/frozen_task.json'),task)
eq('identity/acceptance',physical['acceptance'],profile)
eq('identity/ledger_hash',js(RUN/'ledger.json')['identity']['identity_hash'],digest(ident))
domains['source_identity']={'production_files':len(source),'zip_members':len(source),'bound_input_files':len(binding),'old_raw_interpreted':0,'production_commit':info['producer_head']}
prefix=npz(ROOT/sel['units'][0]['prefix_file'])
orig=npz(UNIT/'execution/initial_state.npz')['values'];origmeta=js(UNIT/'execution/initial_state.json')
eq('identity/initial_state_digest',digest({'values':orig,'model_hash':origmeta['model_hash'],'state_mask':origmeta['state_mask']}),origmeta['state_id'])
close('identity/full_root_prefix_equals_execution_initial',prefix['state'][0],orig,0,0)
close('identity/planner_root',prefix['q'][0],q0,0,0)
eq('identity/task_initial',task['contract']['initial_state_id'],origmeta['state_id'])
eq('identity/scene_model',task['contract']['robot_hash'],origmeta['model_hash'])
audit=js(RUN/'scenes/capture_5_obstacles/initial_audit.json')
eq('identity/origin_complete_state_hash',digest(orig),audit['origin']['complete_state_hash'])
check('identity/scene_transfer_all_checks',all(audit['scene_transfer']['checks'].values()))
eq('identity/scene_five_objects',len(sel['entries'][0]['scene']['objects']),5)
eq('identity/scene_pairs',audit['initial_geometry']['pair_count'],3250)
eq('identity/pair_policy',report['pair_policy_hash'],task['contract']['pair_policy_hash'])
pi=js(UNIT/'prefix_identity.json');check('identity/prefix_declared_checks',all(pi['checks'].values()))
close('prefix/parents',prefix['parent'],np.arange(len(prefix['q']))-1,0,0)
close('prefix/source_node_ids',prefix['source_node_id'],sel['units'][0]['source_node_chain'],0,0)
check('prefix/time_and_duration',np.all(np.diff(prefix['elapsed'])>0) and prefix['elapsed'][0]==0 and prefix['elapsed'][-1]==26)
close('prefix/durations',np.diff(prefix['elapsed']),prefix['duration'][1:],1e-12,0)
stats=js(ROOT/'ff_prior_contact_steer/evidence/statistics.json')
best=min(stats['unit_rows'],key=lambda x:(x['new_best_Q'],x['unit'],x['best_node']))
eq('source/deterministic_selection',(best['unit'],best['best_node'],best['new_best_Q']),('u02',149,sel['units'][0]['source_Q']))

K=np.r_[np.zeros(4),np.arange(1,21)/21,np.ones(4)]
def control(z):return np.vstack([np.repeat(q0[None],3,axis=0),np.asarray(z).reshape(19,17)[:18],np.repeat(np.asarray(z).reshape(19,17)[-1:],3,axis=0)])
def curve(z,o=0):return BSpline(K,control(z),3,extrapolate=False).derivative(o)
def value(z,t,o=0):
    t=np.asarray(t);v=curve(z,o)(np.clip(t/27,0,1))/27**o
    return np.where(((t<0)|(t>27))[...,None],0,v) if o else v
def poseerr(a,b):
    a=np.asarray(a);b=np.asarray(b);p=np.linalg.norm(a[...,:3,3]-b[...,:3,3],axis=-1)
    R=np.swapaxes(b[...,:3,:3],-1,-2)@a[...,:3,:3]
    r=Rotation.from_matrix(R.reshape(-1,3,3)).magnitude().reshape(p.shape)
    return p,r
def posevec(a):
    a=np.asarray(a);p=((a[...,:3,3]-goals[:,:3,3])/.005).reshape(a.shape[:-3]+(6,))
    R=np.swapaxes(goals[:,:3,:3],-1,-2)@a[...,:3,:3]
    r=(Rotation.from_matrix(R.reshape(-1,3,3)).as_rotvec()/np.deg2rad(1)).reshape(p.shape)
    return np.concatenate([p,r],axis=-1)
def merit(g,d):
    v=np.minimum((np.asarray(d)-.0002)/.003,0.)
    return .05*float(np.asarray(g)@np.asarray(g))+(float(np.mean(v*v)+.25*np.max(v*v)) if len(v) else 0.)
def metrics(a,meta):
    groups=np.linalg.norm(a['goal'].reshape(4,3),axis=1)
    close(meta+'/goal_vector',a['goal'],posevec(a['ee'][-1]),1e-12,1e-10)
    return {'Q':float(max(groups)),'goal_groups':groups,'minimum_clearance_m':float(min(a['minima'])),'merit':merit(a['goal'],a['minima'])}
def matchmet(name,got,expected):
    for k,v in got.items():close(name+'/'+k,v,expected[k],1e-12,1e-9)
B=np.zeros((67,17));B[:60,:10]=np.kron(np.eye(5),np.array([[0,1],[1,0],[1,0],[0,1],[0,1],[1,0],[1,0],[0,1],[0,1],[1,0],[1,0],[0,1]])/6);B[60:,10:]=np.eye(7);Bp=np.linalg.pinv(B)
names=physical['identity']['robot']['low_level_joint_names'];imap=physical['identity']['joint_index_map']
qi=np.array([imap[n]['qpos'] for n in names]);vi=np.array([imap[n]['dof'] for n in names]);ai=np.array([imap[n]['actuator'] for n in names])
urdf=ET.fromstring(raw(ROOT/'dual_arm_space_robot_2026/urdf/dual_arm_space_robot_2026.urdf'))
joints={x.attrib['name']:x for x in urdf.findall('joint')};limited=np.array([joints[n].attrib['type']=='revolute' for n in names])
ranges=np.array([[float(joints[n].find('limit').get('lower','0')),float(joints[n].find('limit').get('upper','0'))] if joints[n].find('limit') is not None else [0.,0.] for n in names])
vlim=np.r_[np.full(10,.8),np.full(7,1.2)];alim=np.r_[np.full(10,2.5),np.full(7,4.)];tlim=np.r_[np.full(60,4.),np.full(7,80.)]
lower=np.asarray(physical['identity']['planner_lower']);upper=np.asarray(physical['identity']['planner_upper'])
maps=[];offsets=[]
for o in (0,1,2):
    c=curve(np.zeros((19,17)),o);offset=c.c[:len(c.t)-c.k-1]/27**o;offsets.append(offset);columns=[]
    for j in range(19):
        z=np.zeros((19,17));z[j]=1;c=curve(z,o);columns.append((c.c[:len(c.t)-c.k-1]/27**o-offset)[:,0])
    maps.append(np.array(columns).T)
A=[];lo=[];hi=[]
for o,lim in ((1,vlim),(2,alim)):
    A.append(np.kron(maps[o],np.eye(17)));bounds=np.broadcast_to(.995*lim,offsets[o].shape)
    lo.append((-bounds-offsets[o]).ravel());hi.append((bounds-offsets[o]).ravel())
A.append(np.kron(maps[0],B[limited]));offset=offsets[0]@B[limited].T
lo.append((ranges[limited,0]+1e-10-offset).ravel());hi.append((ranges[limited,1]-1e-10-offset).ravel())
A=np.vstack(A);lo=np.concatenate(lo);hi=np.concatenate(hi);keep=np.any(A!=0,axis=1);A=A[keep];lo=lo[keep];hi=hi[keep]
close('motion/A_full',fit['A'],A,1e-13,1e-12);close('motion/lo_full',fit['lo'],lo);close('motion/hi_full',fit['hi'],hi)
close('motion/variable_lower',fit['variable_lower'],np.tile(lower+1e-10,19));close('motion/variable_upper',fit['variable_upper'],np.tile(upper-1e-10,19))
motion_checked=[]
def motion(z,name):
    x=np.asarray(z).ravel();y=A@x;c=control(z);vl=[]
    for o,lim in ((1,vlim),(2,alim)):
        cc=curve(z,o);polygon=cc.c[:len(cc.t)-cc.k-1]/27**o;vl.append(float(np.max(abs(polygon)/lim)))
    residual=float(max(np.max(lo-y),np.max(y-hi),np.max(fit['variable_lower']-x),np.max(x-fit['variable_upper'])))
    boundary=max(np.max(abs(value(z,t,o))) for t in (0.,27.) for o in (1,2))
    valid=bool(np.isfinite(x).all() and residual<=1e-9 and np.all(c>=lower) and np.all(c<=upper) and np.all(c@B[limited].T>=ranges[limited,0]) and np.all(c@B[limited].T<=ranges[limited,1]) and max(vl)<=1+1e-12 and boundary<1e-10)
    check('motion/'+name,valid,linear_rows=len(A),linear_violation=residual,derivative_limit_ratios=vl,boundary_max=float(boundary))
    motion_checked.append(name);return valid
motion(fit['Z'],'fit')
close('fit/times',fit['times'],np.linspace(0,27,545),0,0)
desired=[]
for t in fit['times']:
    row=prefix['q'][-1]
    for k in range(1,len(prefix['q'])):
        if prefix['elapsed'][k-1]<=t<=prefix['elapsed'][k]:
            s=(t-prefix['elapsed'][k-1])/prefix['duration'][k];row=prefix['q'][k-1]+(10*s**3-15*s**4+6*s**5)*(prefix['q'][k]-prefix['q'][k-1]);break
    desired.append(row)
close('fit/desired_all_545x17',fit['desired'],desired,1e-14,1e-12);close('fit/fitted_all_545x17',fit['fitted'],value(fit['Z'],fit['times']))
def fitloss(z):return float(np.mean((value(z,fit['times'])-fit['desired'])**2)+1e-4*np.mean(np.diff(z,n=2,axis=0)**2))
close('fit/returned_loss',fitloss(fit['Z']),fitmeta['returned_loss']);close('fit/initial_loss',fitloss(np.tile(q0,(19,1))),fitmeta['initial_loss'])

def select(near):
    ranked=sorted(range(len(near)),key=lambda i:(near[i,3],*near[i,:3]));chosen=[];why=[];pairs=set();bins=set()
    def add(i,w):
        if i not in chosen and len(chosen)<240:chosen.append(i);why.append(w)
    for i in ranked:
        p=tuple(near[i,1:3].astype(int))
        if p not in pairs:add(i,'worst_for_geometry_pair');pairs.add(p)
    for i in ranked:
        b=min(63,int(near[i,0]*64))
        if b not in bins:add(i,'worst_for_phase_bin');bins.add(b)
    for i in ranked:add(i,'remaining_by_distance')
    order=np.argsort(chosen);return np.asarray(chosen,dtype=int)[order],[why[i] for i in order]
dense={};dense_totals={'evaluations':0,'states':0,'buffer_records':0,'selected':0,'omitted':0,'grid_calls':0,'intervals':0}
for path in [UNIT/'restore/initial.npz']+sorted((UNIT/'restore/iter').glob('*/trial*.npz')):
    a=npz(path);j=js(path.with_suffix('.json'));name=str(path.relative_to(UNIT));n=len(a['phase'])-1
    close(name+'/phases',a['phase'],np.linspace(0,1,n+1),0,0)
    close(name+'/root',a['qpos'][0],prefix['qpos'][0],1e-14,0)
    close(name+'/joint_spline_full',a['qpos'][:,qi],value(a['Z'],a['phase']*27)@B.T,1e-12,1e-12)
    close(name+'/joint_velocity_full',a['velocity'][:,vi],value(a['Z'],a['phase']*27,1)@B.T,1e-12,1e-12)
    close(name+'/base',a['base'],a['qpos'][:,:7],0,0)
    m=metrics(a,name);matchmet(name+'/metrics',m,j);near=a['near'];si,reason=select(near)
    close(name+'/selected_all',a['selected'],si,0,0);eq(name+'/selection_reasons',j['selected_reasons'],reason)
    close(name+'/omitted_all',a['omitted'],np.setdiff1d(np.arange(len(near)),si),0,0)
    check(name+'/buffer_domain',np.all(near[:,3]<.004) and np.all(near[:,1:3]==near[:,1:3].astype(int)) and np.all((near[:,1:3]>=0)&(near[:,1:3]<159)) and len(np.unique(near[:,:3],axis=0))==len(near))
    nearidx=np.searchsorted(a['phase'],near[:,0]);close(name+'/buffer_phases',a['phase'][nearidx],near[:,0],0,0)
    buffer_minima=np.full(n+1,np.inf);np.minimum.at(buffer_minima,nearidx,near[:,3]);has=np.isfinite(buffer_minima)
    close(name+'/all_buffer_phase_minima',buffer_minima[has],a['minima'][has],1e-13,1e-12)
    check(name+'/all_empty_buffer_phases',np.all(a['minima'][~has]>=.004),phases_with_buffer=int(sum(has)),phases_without_buffer=int(sum(~has)))
    check(name+'/minimum_pairs_count',len(j['minimum_pairs'])==n+1)
    eq(name+'/policy',j['pair_policy_hash'],task['contract']['pair_policy_hash'])
    eq(name+'/cache_key',j['key'],digest({'z':a['Z'],'initial':origmeta['state_id'],'task':task['task_hash'],'model':origmeta['model_hash'],'policy':task['contract']['pair_policy_hash'],'protocol':P}))
    eq(name+'/grid_prefix',[x['intervals'] for x in j['grid_history']],P['dense_grids'][:len(j['grid_history'])])
    check(name+'/grid_stopping',all(x['geometry_step_m']>.002 for x in j['grid_history'][:-1]) and j['grid_history'][-1]['geometry_step_m']<=.002)
    for row in j['grid_history']:
        num=row['numerical'];check(name+f"/numerical/{row['intervals']}",num['solve_residual']<1e-10 and num['robot_momentum_residual']<1e-10 and num['quaternion_norm_error']<1e-12)
    motion(a['Z'],name);dense[str(path)]=(a,j,m)
    dense_totals['evaluations']+=1;dense_totals['states']+=n+1;dense_totals['buffer_records']+=len(near);dense_totals['selected']+=len(si);dense_totals['omitted']+=len(a['omitted']);dense_totals['grid_calls']+=len(j['grid_history']);dense_totals['intervals']+=sum(x['intervals'] for x in j['grid_history'])
domains['dense']=dense_totals
print('dense checked',dense_totals,flush=True)
current=dense[str(UNIT/'restore/initial.npz')];close('restore/fit_starts_initial',current[0]['Z'],fit['Z'],0,0)
radius=.0375;rows=[];sparse_calls=0;sparse_intervals=0
for pd in sorted((UNIT/'restore/iter').iterdir()):
    if not pd.is_dir():continue
    j=npz(pd/'jacobian.npz');rec=js(pd/'record.json');probes=js(pd/'probes.json');name='iter/'+pd.name;before,meta,met=current
    check(name+'/full323_shapes',j['jg'].shape==(12,323) and j['jd'].shape==(len(j['active']),323) and j['perturb_goal'].shape==(323,12) and j['perturb_distances'].shape==(323,len(j['active'])) and j['perturb_active_qpos'].shape==(323,len(j['active']),81))
    close(name+'/Z_chain',j['Z'],before['Z'],0,0);close(name+'/active_identity',j['active'],before['near'][before['selected'],:3],0,0)
    close(name+'/goal_ee',j['goal'],posevec(j['ee']));close(name+'/perturb_goal_ee_all',j['perturb_goal'],posevec(j['perturb_ee']))
    x=j['Z'].ravel();h=np.where(fit['variable_upper']-x>=x-fit['variable_lower'],.0001,-.0001)
    h=np.sign(h)*np.minimum(abs(h),np.maximum(fit['variable_upper']-x,x-fit['variable_lower'])*.5)
    close(name+'/all323_steps',j['steps'],h,0,0)
    close(name+'/all_jg',j['jg'],(j['perturb_goal']-j['goal']).T/h,0,0)
    close(name+'/all_jd',j['jd'],(j['perturb_distances']-j['distances']).T/h,0,0)
    # No new kinematics: test every archived active/terminal joint configuration against the analytic spline.
    close(name+'/center_active_joint_values',j['active_qpos'][:,qi],value(j['Z'],j['active'][:,0]*27)@B.T)
    close(name+'/center_terminal_joint_values',j['terminal_qpos'][qi],j['Z'][-1]@B.T)
    perr=0.;terr=0.;quaterr=0.
    for col in range(323):
        zz=j['Z'].copy();zz.ravel()[col]+=h[col]
        perr=max(perr,float(np.max(abs(j['perturb_active_qpos'][col][:,qi]-value(zz,j['active'][:,0]*27)@B.T),initial=0)))
        terr=max(terr,float(np.max(abs(j['perturb_terminal_qpos'][col,qi]-zz[-1]@B.T))))
        quaterr=max(quaterr,float(np.max(abs(np.linalg.norm(j['perturb_active_qpos'][col,:,3:7],axis=-1)-1),initial=0)))
    check(name+'/all323_perturb_joint_values',perr<1e-11 and terr<1e-11,active_max=perr,terminal_max=terr,base_quaternion_norm_error=quaterr)
    close(name+'/radius',rec['radius_rad'],radius,0,0)
    matchmet(name+'/before',met,rec['before'])
    ag=rec['sparse_agreement'];eq(name+'/sparse_match_attempt_count',len(ag),1)
    phase=np.unique(np.r_[np.linspace(0,1,513),j['active'][:,0]])
    close(name+'/sparse_full_phase_union',j['phase'],phase,0,0)
    ep,er=poseerr(j['ee'],before['ee'][-1]);dd=float(np.max(abs(j['distances']-before['near'][before['selected'],3]),initial=0))
    close(name+'/sparse_position',ag[-1]['position_m'],ep);close(name+'/sparse_rotation',ag[-1]['rotation_rad'],er);close(name+'/sparse_distance',ag[-1]['distance_m'],dd)
    check(name+'/sparse_gate',max(ep)<=.001 and max(er)<=np.deg2rad(.2) and dd<=.001 and ag[-1]['passed'])
    eq(name+'/probe_count',len(probes['observations']),4);eq(name+'/no_probe_pass_threshold',probes['acceptance_threshold'],None)
    coords=[int(np.argmax(np.linalg.norm(j['jg'],axis=0))),int(np.argmax(np.linalg.norm(j['jd'],axis=0))) if len(j['jd']) else 161]
    d=np.sin(np.arange(323)+.37);d/=np.linalg.norm(d);diag=[]
    for k,p in enumerate(probes['observations']):
        zz=j['Z'].copy();g=np.asarray(p['goal']);dist=np.asarray(p['distances'])
        if k<2:
            eq(name+f'/probe{k}_coordinate',p['coordinate'],coords[k]);close(name+f'/probe{k}_halfstep',p['step'],h[coords[k]]/2,0,0);zz.ravel()[coords[k]]+=p['step'];delta=np.eye(323)[coords[k]]*p['step']
        else:
            close(name+f'/probe{k}_direction',p['direction'],d,0,0);eq(name+f'/probe{k}_sign',p['sign'],[-1.,1.][k-2]);delta=p['sign']*p['step']*d;zz+=delta.reshape(19,17)
        close(name+f'/probe{k}_goal_pose',g,posevec(p['ee']))
        close(name+f'/probe{k}_active_joints',np.asarray(p['active_qpos'])[:,qi],value(zz,j['active'][:,0]*27)@B.T)
        close(name+f'/probe{k}_terminal_joints',np.asarray(p['terminal_qpos'])[qi],zz[-1]@B.T)
        num=p['numerical'];check(name+f'/probe{k}_numerical',num['solve_residual']<1e-10 and num['robot_momentum_residual']<1e-10 and num['quaternion_norm_error']<1e-12)
        diag.append({'kind':p['kind'],'goal_linearization_abs_max':float(np.max(abs(g-(j['goal']+j['jg']@delta)))),'distance_linearization_abs_max_m':float(np.max(abs(dist-(j['distances']+j['jd']@delta)),initial=0))})
    derivatives.append({'iteration':rec['iteration'],'observations':diag,'pass_gate':None})
    sol=rec['solver'];u=np.asarray(sol['returned_u']);step=radius*u;close(name+'/returned_step',j['step'],step,0,0)
    check(name+'/trust_bound',np.max(abs(step))<=radius+1e-12);motion(j['Z']+step.reshape(19,17),name+'/returned_solver')
    normal=max(1.,merit(j['goal'],j['distances']));close(name+'/normalizer',sol['normalizer'],normal)
    objective=(merit(j['goal']+j['jg']@step,j['distances']+j['jd']@step)+1e-4*float(u@u)/323)/normal
    close(name+'/returned_objective',sol['returned_objective'],objective,1e-12,1e-9)
    close(name+'/initial_objective',sol['initial_objective'],merit(j['goal'],j['distances'])/normal)
    G=j['jg']*radius;D=j['jd']*radius/.003;v=np.minimum((j['distances']-.0002)/.003+D@u,0);g=j['goal']+G@u
    grad=2*.05*G.T@g+2e-4*u/323
    if len(v):k=int(np.argmax(v*v));grad+=2*D.T@v/len(v)+.5*v[k]*D[k]
    close(name+'/returned_gradient_all323',sol['returned_gradient'],grad/normal,1e-11,1e-9)
    check(name+'/solver_cap',sol['iterations']<=180 and sol['callbacks']<=181)
    trials=[];accepted=False
    for k,t in enumerate(rec['trials']):
        eq(name+f'/trial{k}_line_scale',t['scale'],P['line_scales'][k]);z=j['Z']+t['scale']*step.reshape(19,17);valid=motion(z,name+f'/trial{k}')
        if t['status']=='MOTION_REJECTED':check(name+f'/trial{k}_proper_motion_reject',not valid);continue
        a,m,mt=dense[str(pd/f'trial{k}.npz')];close(name+f'/trial{k}_Z',a['Z'],z,0,0);matchmet(name+f'/trial{k}_metrics',mt,t['metrics'])
        predicted=merit(j['goal'],j['distances'])-merit(j['goal']+j['jg']@(t['scale']*step),j['distances']+j['jd']@(t['scale']*step))
        reduction=met['merit']-mt['merit'];ratio=reduction/predicted if predicted>1e-12 else None
        close(name+f'/trial{k}_actual_reduction',t['actual_reduction'],reduction);close(name+f'/trial{k}_predicted_reduction',t['predicted_reduction'],predicted);close(name+f'/trial{k}_ratio',t['ratio'],ratio)
        ok=reduction>1e-10 and ratio is not None and ratio>=.1;eq(name+f'/trial{k}_accept',t['accepted'],ok)
        trials.append({'scale':t['scale'],'accepted':ok,'actual_reduction':reduction,'predicted_reduction':predicted,'ratio':ratio})
        if ok:check(name+'/stop_at_first_accept',k==len(rec['trials'])-1);current=(a,m,mt);accepted=True
    eq(name+'/accepted',rec['accepted'],accepted);close(name+'/next_Z_chain',j['next_Z'],current[0]['Z'],0,0);matchmet(name+'/after',current[2],rec['after'])
    nr=max(.001171875,radius/2) if not accepted or rec['trials'][-1]['ratio']<.25 or rec['trials'][-1]['scale']<1 else (min(.15,2*radius) if rec['trials'][-1]['ratio']>.75 and max(abs(step))>=.9*radius else radius)
    close(name+'/next_radius',rec['next_radius_rad'],nr,0,0);radius=nr
    sparse_calls+=len(ag)+323+4;sparse_intervals+=sum(x['intervals'] for x in ag)+(323+4)*(len(j['phase'])-1)
    rows.append({'iteration':int(pd.name),'active_rows':len(j['active']),'jacobian_elements':j['jg'].size+j['jd'].size,'perturb_state_elements':j['perturb_active_qpos'].size+j['perturb_terminal_qpos'].size,'sparse_intervals':len(j['phase'])-1,'solver_success':sol['success'],'solver_iterations':sol['iterations'],'radius':rec['radius_rad'],'next_radius':nr,'trials':trials,'before_merit':met['merit'],'after_merit':current[2]['merit']})
    print('iteration',pd.name,'checked',flush=True)
rest=js(UNIT/'restore/result.json');close('restore/final_Z',rest['Z'],current[0]['Z'],0,0);matchmet('restore/final_metrics',current[2],rest)
check('restore/stop_gate',rest['stop']=='RESTORATION_TARGET_REACHED' and rest['Q']<=.8 and rest['minimum_clearance_m']>=.0002 and rest['geometry_step_m']<=.002)
eq('restore/iterations',rest['iterations'],len(rows))
domains['restore']={'iterations':len(rows),'finite_difference_columns':323*len(rows),'jacobian_elements':sum(x['jacobian_elements'] for x in rows),'perturb_state_elements':sum(x['perturb_state_elements'] for x in rows),'probes':4*len(rows),'sparse_calls':sparse_calls,'sparse_model_intervals':sparse_intervals,'rows':rows,'derivative_observations':derivatives}

print('validation/execution',flush=True)
grid_paths=sorted((UNIT/'validated').glob('grid_*.npz'),key=lambda p:int(p.stem.split('_')[1]))
grid_paths=[p for p in grid_paths if 'geometry' not in p.name]
grids={};geoscans=0;validation_numerical=[]
for p in grid_paths:
    n=int(p.stem.split('_')[1]);a=npz(p);grids[n]=a
    num=js(p.with_name(p.stem+'_numerical.json'));validation_numerical.append(num)
    check(f'validation/{n}/numerical',num['solve_residual']<1e-10 and num['robot_momentum_residual']<1e-10 and num['quaternion_norm_error']<1e-12)
    close(f'validation/{n}/phase',a['phase'],np.linspace(0,1,n+1),0,0)
    close(f'validation/{n}/root',a['qpos'][0],prefix['qpos'][0],1e-14,0)
    close(f'validation/{n}/joint_reference',a['qpos'][:,qi],value(rest['Z'],a['phase']*27)@B.T)
    close(f'validation/{n}/joint_velocity',a['velocity'][:,vi],value(rest['Z'],a['phase']*27,1)@B.T)
    close(f'validation/{n}/base',a['base'],a['qpos'][:,:7],0,0)
    check(f'validation/{n}/quaternion_norm',np.max(abs(np.linalg.norm(a['qpos'][:,3:7],axis=1)-1))<1e-12)
    if n>256:
        geo=npz(p.with_name(p.stem+'_geometry_scans.npz'));gj=js(p.with_name(p.stem+'_geometry_scans.json'));gmeta=next(x for x in validation['validation_grids'] if x['intervals']==n)
        close(f'validation/{n}/scan_state_order',geo['qpos'],np.r_[a['qpos'][::2],a['qpos'][1::2]],0,0)
        close(f'validation/{n}/scan_minima_json_npz',geo['minima'],[x['minimum_clearance_m'] for x in gj],0,0)
        cut=n//2+1;mc=float(min(geo['minima'][:cut]));mm=float(min(geo['minima'][cut:]));minimum=min(mc,mm)
        for k,v in [('minimum_clearance_m',minimum),('refinement_difference_m',abs(mc-minimum))]:close(f'validation/{n}/geometry/{k}',gmeta['geometry'][k],v)
        for role,values in [('coarse',geo['minima'][:cut]),('midpoints',geo['minima'][cut:])]:
            close(f'validation/{n}/{role}_minimum',gmeta['geometry'][role]['minimum_clearance_m'],min(values));eq(f'validation/{n}/{role}_count',gmeta['geometry'][role]['checked_states'],len(values))
        check(f'validation/{n}/nonpenetration',np.all(geo['minima']>=0) and all(x['threshold_m']==0 for x in gj))
        p_err,r_err=poseerr(a['ee'][-1],grids[n//2]['ee'][-1]);cmp=validation['validation']['numerical']['comparisons'][len(grids)-2]
        close(f'validation/{n}/endpoint_position_comparison',p_err,cmp['terminal_position_m']);close(f'validation/{n}/endpoint_rotation_comparison',r_err,cmp['terminal_rotation_rad'])
        bodymax=float(np.max(np.linalg.norm(a['body'][::2]-grids[n//2]['body'],axis=-1)))
        geommax=float(np.max(np.linalg.norm(a['geom'][::2]-grids[n//2]['geom'],axis=-1)))
        check(f'validation/{n}/saved_extent_consistency_lower_bound',max(bodymax,geommax)<=cmp['aligned_body_extent_m']+1e-12 and cmp['aligned_body_extent_m']<=.001,body_center_max=bodymax,geom_center_max=geommax)
        geoscans+=len(gj)
pred=npz(UNIT/'validated/prediction.npz');finalgrid=grids[max(grids)]
for k in pred:close('validation/final_prediction/'+k,pred[k],finalgrid[k],0,0)
for k in ('qpos','ee','base','phase'):close('validation/restore_agrees/'+k,current[0][k],pred[k],0,0)
close('validation/Z',validation['Z'],rest['Z'],0,0)
refhash=digest({'C':control(rest['Z']),'knots':K,'degree':3,'T':27.,'hold':.5})
for name,h in [('restoration',rest['reference_hash']),('validation',validation['reference_hash']),('execution',report['metadata']['reference_hash'])]:eq('identity/reference_hash/'+name,h,refhash)
p_err,r_err=poseerr(pred['ee'][-1],goals)
close('validation/final_goal_position',p_err,validation['validation']['frozen_goal_position_error_m']);close('validation/final_goal_rotation',r_err,validation['validation']['frozen_goal_rotation_error_rad'])
close('validation/score',validation['score'],sum((p_err/.005)**2)+sum((r_err/np.deg2rad(1))**2))
check('validation/final_gates',validation['eligible'] and validation['validation']['validation_level']=='MODEL_VALID' and all(validation['validation']['gates'].values()) and max(p_err)<=.005 and max(r_err)<=np.deg2rad(1))
motion(rest['Z'],'final_validated')
domains['validation']={'grids':list(grids),'grid_states':sum(len(a['phase']) for a in grids.values()),'geometry_scan_states':geoscans,'saved_numerical':validation_numerical,'final_model_minimum_clearance_m':validation['validation']['geometry']['minimum_clearance_m'],'final_model_geometry_step_m':validation['validation']['geometry']['maximum_geometry_step_m']}

a=npz(UNIT/'execution/native_trace.npz');b=npz(UNIT/'execution/replay_trace.npz');t=a['time']-a['time'][0];dt=.002
replay_domains={}
for k in b:
    close('execution/full_replay/'+k,a[k],b[k],0,0)
    replay_domains[k]={'shape':a[k].shape,'elements':a[k].size,'max_abs_difference':float(np.max(abs(a[k]-b[k]),initial=0))}
eq('execution/state_shape',a['integration'].shape,(13751,836));eq('execution/control_shape',a['ctrl'].shape,(13750,67))
close('execution/first_integration_equals_original',a['integration'][0],orig,1e-12,0)
close('execution/time_integration',a['integration'][:,0],a['time'],0,0)
close('execution/qpos_integration',a['integration'][:,1:82],a['qpos'],0,0)
close('execution/qvel_integration',a['integration'][:,82:161],a['qvel'],0,0)
close('execution/warmstart_integration',a['integration'][:,161:240],a['qacc_warmstart'],0,0)
close('execution/control_applied_alignment',a['integration'][1:,240:307],a['ctrl'],0,0)
close('execution/initial_ctrl',a['integration'][0,240:307],np.zeros(67),0,0)
close('execution/all_generalized_and_body_wrenches_zero',a['integration'][:,307:],np.zeros((len(t),529)),0,0)
close('execution/time_step',np.diff(t),np.full(len(t)-1,dt),1e-12,0);close('execution/final_time',t[-1],27.5,1e-9,0)
for k in ('initial_state_id','scene_model_hash'):eq('execution/metadata_'+k,report[k],report['metadata'][k])
eq('execution/control_hash',digest(a['ctrl']),report['ctrl_hash']);eq('execution/reference_hash',report['metadata']['reference_hash'],refhash)
close('execution/base',a['base'],a['qpos'][:,:7],0,0)
q=a['qpos'][:,qi]@Bp.T;v=a['qvel'][:,vi]@Bp.T;acc=np.diff(v,axis=0)/dt;lv=a['qvel'][:,vi];la=np.diff(lv,axis=0)/dt;rq=value(rest['Z'],t)
close('execution/actual17',a['q_actual_17'],q,1e-13,1e-12)
man=a['qpos'][:,qi]-q@B.T;close('execution/manifold_all',a['manifold_residual'],man,1e-13,1e-12)
close('execution/manifold_max',report['manifold_max_rad'],np.max(abs(man)),1e-12,0)
close('execution/tracking_max',report['tracking_max_rad'],np.max(abs(q-rq)),1e-12,0)
for name,o in [('reference_q',0),('reference_qd',1),('reference_qdd',2)]:close('execution/'+name,a[name],value(rest['Z'],np.arange(len(a['ctrl']))*dt,o),1e-13,1e-12)
check('execution/joint_ranges_all',np.all(q>=lower) and np.all(q<=upper) and np.all(rq>=lower) and np.all(rq<=upper) and np.all(a['qpos'][:,qi][:,limited]>=ranges[limited,0]) and np.all(a['qpos'][:,qi][:,limited]<=ranges[limited,1]))
ratios={'actual_planner_velocity':float(np.max(abs(v)/vlim)),'actual_planner_acceleration':float(np.max(abs(acc)/alim)),'actual_low_velocity':float(np.max(abs(lv)/(abs(B)@vlim))),'actual_low_acceleration':float(np.max(abs(la)/(abs(B)@alim))),'reference_velocity':float(np.max(abs(value(rest['Z'],t,1))/vlim)),'reference_acceleration':float(np.max(abs(value(rest['Z'],t,2))/alim))}
check('execution/derivative_limits_all',max(ratios.values())<=1,ratios=ratios)
check('execution/raw_torque_all',np.all(abs(a['raw_torque'])<=tlim+1e-12),maximum_ratio=float(np.max(abs(a['raw_torque'])/tlim)))
close('execution/ctrl_matches_raw_clipped',a['ctrl'][:,ai],np.clip(a['raw_torque'],-tlim,tlim),1e-12,0)
pos,rot=poseerr(a['ee'],goals);hold=t>=27-1e-9
for name,got in [('hold_max_position_error_m',np.max(pos[hold],axis=0)),('hold_max_rotation_error_rad',np.max(rot[hold],axis=0)),('final_position_error_m',pos[-1]),('final_rotation_error_rad',rot[-1])]:close('execution/'+name,report[name],got,1e-12,1e-10)
check('execution/hold_gate_all',np.all(pos[hold]<=.005) and np.all(rot[hold]<=np.deg2rad(1)),hold_states=int(sum(hold)))
peak=np.array([np.max(np.linalg.norm(a['base_generalized_momentum'][:,s],axis=1)) for s in (slice(0,3),slice(3,6))]);close('execution/momentum_peak',report['momentum_peak_linear_angular'],peak)
close('execution/momentum_component_max',report['zero_momentum_max_residual'],np.max(abs(a['base_generalized_momentum'])))
threshold=1e-5+.002*np.asarray(report['momentum_exchange_scale_linear_angular']);close('execution/momentum_threshold_formula',report['momentum_threshold_linear_angular'],threshold)
check('execution/momentum_gate_recorded_denominator',np.all(peak<=threshold),note='exchange-scale raw mass blocks not saved; denominator source verified, not independently rebuilt')
close('execution/base_max',report['maximum_base_translation_m'],np.max(np.linalg.norm(a['base'][:,:3]-a['base'][0,:3],axis=1)))
close('execution/target_satellite_qpos_constant',a['qpos'][:,74:],np.broadcast_to(a['qpos'][0,74:],a['qpos'][:,74:].shape),1e-12,0)
close('execution/target_satellite_velocity_zero',a['qvel'][:,73:],np.zeros_like(a['qvel'][:,73:]),1e-12,0)
geo=npz(UNIT/'execution/geometry_scans.npz');gjson=js(UNIT/'execution/geometry_scans.json');eq('execution/geometry_count',len(geo['qpos']),27501)
close('execution/geometry_json_minima',geo['minima'],[x['minimum_clearance_m'] for x in gjson],0,0)
expected=np.empty_like(geo['qpos']);expected[0]=a['qpos'][0];expected[1::2]=a['qpos'][1:]
mid=(a['qpos'][:-1]+a['qpos'][1:])/2
for sl in (slice(3,7),slice(77,81)):
    qa=a['qpos'][:-1,sl];qb=a['qpos'][1:,sl].copy();qb[np.sum(qa*qb,axis=1)<0]*=-1;qm=qa+qb;qm/=np.linalg.norm(qm,axis=1,keepdims=True);mid[:,sl]=qm
expected[2::2]=mid;close('execution/all_geometry_scan_state_identity',geo['qpos'],expected,1e-13,1e-12)
correct_rows=[(0,0.)]+[(v,f) for k in range(1,len(t)) for v,f in ((k,0.),(k-1,.5))]
check('execution/all_geometry_scan_index_fraction',all((v['state'],v['fraction'])==e for v,e in zip(gjson,correct_rows)))
indices=np.r_[0,np.arange(1,len(geo['minima']),2)];close('execution/native_clearance_matches_scan',a['clearance_m'],geo['minima'][indices],0,0)
close('execution/whole_scan_minimum',report['minimum_clearance_m'],min(geo['minima']),0,0)
eq('execution/minimum_pair_identity',report['minimum_pair'],gjson[int(np.argmin(geo['minima']))]['pair'])
check('execution/all_observed_clearance',np.all(geo['minima']>=0));check('execution/15_report_gates',len(report['gates'])==15 and all(report['gates'].values()))
domains['execution']={'native_states':len(t),'control_steps':len(a['ctrl']),'hold_states':int(sum(hold)),'replay_full_domains':replay_domains,'geometry_scan_states':len(gjson),'minimum_clearance_m':float(min(geo['minima'])),'maximum_geometry_step_reported_m':report['maximum_geometry_step_m'],'hold_position_m':np.max(pos[hold],axis=0),'hold_rotation_rad':np.max(rot[hold],axis=0),'derivative_ratios':ratios,'raw_torque_max_ratio':float(np.max(abs(a['raw_torque'])/tlim)),'momentum_peak':peak,'momentum_threshold':threshold,'final_state_digest':digest(a['integration'][-1]),'final_time_s':float(a['time'][-1]),'final_state_all_fields_equal_replay':bool(np.array_equal(a['integration'][-1],b['integration'][-1])),'satellite_motion':'observed constant physical pose and zero physical velocity'}

budget=js(RUN/'budget.json');ledger=js(RUN/'ledger.json');calls=js(UNIT/'calls.json');summary=js(RUN/'summary.json');status=js(RUN/'status.json')
count=dense_totals['grid_calls']+sparse_calls+len(grids);intervals=dense_totals['intervals']+sparse_intervals+sum(grids)
for name,v in [('budget_reserved',budget['reserved_coupled']),('budget_actual',budget['actual_coupled']),('calls',calls['coupled']),('summary',summary['actual_coupled']),('ledger',ledger['calls']['full27s_323_restoration.coupled'])]:eq('budget/coupled/'+name,count,v)
eq('budget/model_intervals',budget['integration_intervals'],intervals)
for name,v in [('budget',budget['actual_native_and_replay_steps']),('calls',calls['physics_steps']),('summary',summary['new_native_and_replay_steps'])]:eq('budget/native/'+name,v,len(a['ctrl'])*2)
eq('budget/one_rollout',calls['torque_rollout'],1);eq('budget/solver_calls',calls['sqp'],8)
check('budget/hard_limits',count<=14000 and intervals<=35500000 and calls['physics_steps']<=27500 and status['elapsed_s']<=18000)
derivation=js(ROOT/'ff_prior_horizon_restore/evidence/budget_derivation.json');auth=js(ROOT/'ff_prior_horizon_restore/evidence/authorization.json')
eq('budget/iteration_calls_formula',323+3+4+6*6,P['iteration_query_reserve']);eq('budget/iteration_intervals_formula',327*2288+752+1264+2288+6*32256,P['iteration_interval_reserve'])
check('budget/original_D3_limits_with_frozen_future',43773+14000+2000<=60000 and 1990.109+18000+1200<=21540 and 27500+27000<=165000 and 766411502+16*1024**3+2*1024**3+1024**3<=20*1024**3)
check('budget/original_cycle_limits',derivation['cycle_conservative_total']['coupled']<=80000 and derivation['cycle_conservative_total']['worker_s']<=43200 and derivation['cycle_conservative_total']['native']<=312500 and derivation['cycle_conservative_total']['bytes']<=40*1024**3)
check('scope/no_new_network_train_TEST',summary['new_network_samples']==0 and summary['training_updates']==0 and summary['TEST'] is False and sel['source_network_has_obstacle_condition'] is False and sel['units'][0]['proposal_seed']==20264627)
domains['budget']={'reconstructed_coupled':count,'reconstructed_model_reserved_intervals':intervals,'native_plus_replay_steps':len(a['ctrl'])*2,'wall_s':status['elapsed_s'],'ledger_unit_wall_s':ledger['wall_s'],'future_dynamic_original_reservation':{'coupled':2000,'worker_s':1200,'native':27000,'bytes':2*1024**3},'unused_static_quota_not_authorization_for_extra_candidates':True}

print('media/hash/resume',flush=True)
vis=UNIT/'visuals';man=js(vis/'manifest.json');eq('media/native_hash',man['source_native_hash'],sha(UNIT/'execution/native_trace.npz'))
for name,v in man['files'].items():eq('media/hash/'+name,sha(touch(vis/name)),v['sha256']);eq('media/size/'+name,(vis/name).stat().st_size,v['bytes'])
html=raw(vis/'index.html').decode('utf-8');payload=json.loads(html.split('const D=',1)[1].split(';const c=',1)[0]);ids=np.unique(np.linspace(0,len(t)-1,420).round().astype(int))
for name,expected in [('time',t[ids]),('ee_poses',a['ee'][ids]),('ee',a['ee'][ids,:,:3,3]),('goals',goals[:,:3,3]),('goal_poses',goals),('position_mm',pos[ids]*1000),('clearance_mm',a['clearance_m'][ids]*1000)]:close('media/interactive_payload/'+name,payload[name],expected,0,0)
eq('media/obstacles',payload['obstacles'],sel['entries'][0]['scene']['objects'])
with touch(vis/'timeseries.csv').open(encoding='utf-8',newline='') as f:csvrows=list(csv.DictReader(f))
eq('media/csv_row_count',len(csvrows),len(t))
for name,v in [('time_s',t),('arm0_position_mm',pos[:,0]*1000),('arm1_position_mm',pos[:,1]*1000),('arm0_rotation_deg',np.rad2deg(rot[:,0])),('arm1_rotation_deg',np.rad2deg(rot[:,1])),('clearance_mm',a['clearance_m']*1000)]:close('media/csv/'+name,[float(x[name]) for x in csvrows],v,1e-12,1e-12)
check('media/csv_hold_mask',all((x['hold_gate_active']=='True')==h for x,h in zip(csvrows,hold)))
for s in ['frame(D.ee_poses[i][arm]','frame(D.goal_poses[arm]','T[k][j]','for(let k=1;k<=i;k++)','EE','Goal']:check('media/overlay_code/'+s,s in html)
videos=[]
for p in (vis/'overview.mp4',vis/'top.mp4'):
    probe=subprocess.run(['ffprobe','-v','error','-count_frames','-show_entries','stream=width,height,r_frame_rate,nb_read_frames,duration','-of','json',str(p)],capture_output=True,check=True)
    vp=json.loads(probe.stdout);stream=vp['streams'][0];eq('media/video_frames/'+p.name,int(stream['nb_read_frames']),420);eq('media/video_resolution/'+p.name,(stream['width'],stream['height']),(960,720));eq('media/video_fps/'+p.name,stream['r_frame_rate'],'15/1')
    decode=subprocess.run(['ffmpeg','-v','error','-i',str(p),'-f','null','-'],capture_output=True)
    check('media/full_decode/'+p.name,decode.returncode==0 and not decode.stderr,stderr=decode.stderr.decode(errors='replace'))
    videos.append({'name':p.name,'probe':vp,'full_decode_returncode':decode.returncode})
domains['media']={'manifest_files':len(man['files']),'interactive_frames':len(ids),'csv_rows':len(csvrows),'videos':videos,'recorded_time_s':float(t[-1]),'playback_time_s':28.,'time_note':'420 saved native-state samples, uniform index sampling; 15 fps playback is 28 s, representing 27.5 s physics. No dynamic target was animated.'}
resume=js(ROOT/'ff_prior_horizon_restore/evidence/completed_resume.json');before=js(ROOT/resume['before_manifest']);after=js(ROOT/resume['after_manifest']);eq('resume/before_after',before,after)
for name,v in after.items():eq('resume/current/'+name,sha(touch(RUN/name,'post_resume_hash')),v['sha256']);eq('resume/size/'+name,(RUN/name).stat().st_size,v['bytes'])
for s in ('stdout','stderr'):
    p=ROOT/resume[s+'_file'];eq('resume/'+s+'_hash',sha(touch(p)),resume[s+'_sha256']);eq('resume/'+s+'_size',p.stat().st_size,resume[s+'_bytes'])
eq('resume/no_work',resume['response']['resume'],'NO_WORK_ALREADY_COMPLETE');eq('resume/one_invocation',resume['completed_invocations'],1)
domains['resume']={'files':len(before),'bytes':sum(v['bytes'] for v in before.values()),'one_invocation':True,'identical_before_after_and_current':before==after}
domains['motion']={'linear_matrix_shape':A.shape,'variable_count':323,'checked_references':len(motion_checked),'checks_per_reference':'1923 linear rows plus 323 box bounds, 24 control point native ranges, complete derivative polygons and endpoint boundaries'}
limitations=[
 {'id':'L1','scope':'Dense restoration earlier refinement grids','detail':'Final dense arrays and all grid numerical maxima are saved. Earlier dense-grid qpos/geom arrays are not saved; exact geometry-step and numerical residual recomputation would require prohibited new propagation. Grid order/stopping and source checks verified.'},
 {'id':'L2','scope':'Sparse finite differences','detail':'All 2584 perturbations save full goal/distance outputs, terminal poses and active/terminal qpos. Full nonuniform propagation histories, mass matrices and per-perturbation numerical residual summaries are not saved. Every call is guarded in hashed source; residual maxima cannot be independently rebuilt from archived columns.'},
 {'id':'L3','scope':'Geometry','detail':'All saved per-state minima, near-buffer rows, selected/omitted identities and scan states are completely checked. Per-pair exact distance evaluations/bounding-sphere rejection arrays and compiled geom_rbound are not archived here. Exact distance and geometry-step magnitudes depend on frozen collision/source provenance; no new collision/FK calls were made. No continuous-time certificate is asserted.'},
 {'id':'L4','scope':'Momentum','detail':'All 13751 saved base generalized momentum samples and numerator/threshold arithmetic are checked. Per-state mass blocks and momentum_exchange samples are not saved; exchange-scale denominator is checked against frozen implementation and report only.'},
 {'id':'L5','scope':'Source-only runtime properties','detail':'No hidden state writes and replay independence are supported by complete state/control alignment and hashed tracker/replay code. Recorded histories alone cannot establish arbitrary process instrumentation or independent anti-tamper provenance.'},
 {'id':'L6','scope':'Optimization','detail':'Returned feasible SLSQP steps, objectives, gradients, bounds, all trials and trust updates are checked. Full callback histories are not saved; best-among-all-feasible-iterates is source-verified, not retrospectively proven. SLSQP not rerun; iteration-limit statuses retained.'},
 {'id':'L7','scope':'Media','detail':'All manifest hashes, all 840 video frames decoded, all interactive selected EE/goals/time and 13751 CSV rows checked. Rendered body positions rely on source kinematics; no re-rendering or physics. Parent independently inspects UI screenshots.'},
 {'id':'L8','scope':'Paper-level assurance','detail':'paper/PAPER_CLAIM_AUDIT.json does not exist; downstream paper-writing verdict remains provisional. This is not an obstacle to this bounded engineering handoff.'}
]
gate_evidence={
 'identity':{'mode':'hash_and_array_recomputed','evidence':'task, source archive/current/commit, input hashes, initial/full-root, ctrl/reference IDs'},
 'finite':{'mode':'all_saved_arrays_recomputed','evidence':'every NPZ array read in full; all native/replay arrays finite'},
 'dual_goal_hold':{'mode':'saved_SE3_recomputed','evidence':'all 251 hold states, both arm positions and SO(3) errors'},
 'whole_body_clearance':{'mode':'all_saved_scan_minima_recomputed_plus_source','evidence':'27501 scans; minimum and associated pair/index; exact distances require source provenance, L3'},
 'joint_limits':{'mode':'all_saved_arrays_recomputed','evidence':'13751 states × 17 planner and 67 native coordinates, original URDF limits, spline range'},
 'derivative_limits':{'mode':'all_saved_arrays_recomputed','evidence':'13751 velocities, 13750 finite-difference accelerations, planner/native/ref limits'},
 'raw_torque_limits':{'mode':'all_saved_arrays_recomputed','evidence':'13750 × 67 raw torques, clipping and aligned ctrl'},
 'tracking':{'mode':'all_saved_arrays_recomputed','evidence':'13751 × 17 actual-reference samples'},
 'manifold':{'mode':'all_saved_arrays_recomputed','evidence':'13751 × 67 residuals under frozen 17-to-67 map'},
 'momentum':{'mode':'saved_numerator_and_threshold_formula_plus_source','evidence':'13751 × 6 momentum samples; exchange denominator unavailable as raw arrays, L4'},
 'independent_replay':{'mode':'all_saved_arrays_recomputed_plus_source','evidence':'13751 × 836 integration fields and all separately saved replay fields exactly equal; code applies saved ctrl from fresh initial state'},
 'complete_native_time':{'mode':'all_saved_arrays_recomputed','evidence':'13750 intervals × 0.002 s; complete 27.5 s including 0.5 s hold'},
 'no_external_wrench':{'mode':'all_saved_state_fields_recomputed_plus_source','evidence':'all 79 generalized-force and 75 × 6 body-wrench entries exactly zero at all 13751 states; source checks each step'},
 'no_hidden_reset':{'mode':'source_plus_full_state_time_control_consistency','evidence':'hashed tracker has no qpos/qvel assignment in loop; metadata 0, complete time, integration/control alignment and independent replay'},
 'geometry_step':{'mode':'reported_value_and_source_verified','evidence':'0.0002143638062307468 m <= 0.002 m. All scan qpos/midpoint identities checked; exact geom-motion/rbound arrays absent, L3'}
}
failures=[v for v in checks if not v['passed']]
result={'status':'PASS_WITH_EXPLICIT_EVIDENCE_LIMITS' if not failures else 'FAIL_REQUIRES_RESOLUTION','generated_utc':datetime.now(timezone.utc).isoformat(),'run':'horizon_restore_001','new_physics_calls':0,'new_coupled_calls':0,'new_native_steps':0,'new_replays':0,'new_optimization':0,'new_resume':0,'scope':'all new-run saved domains; no sampling substituted for full checks; inherited raw hashed only','checks_total':len(checks),'checks_passed':len(checks)-len(failures),'unresolved_failures':failures,'domains':domains,'limitations':limitations,'execution_15_gate_evidence':gate_evidence,'all_checks':checks,'array_inventory':inventory}
write(TRACE/'saved_checks.json',result)
# Hash every core result plus every explicitly read dependency. Old raw paths retain a hash-only role.
for p in RUN.rglob('*'):
    if p.is_file():touch(p,'new_run_core_manifest')
for name in ('prompt.md','audit_saved.py'):touch(TRACE/name,'review_definition')
for p in [ROOT/'ff_prior_horizon_restore'/x for x in ('protocol.py','inputs.py','motion.py','solver.py','budget.py','experiment.py','media_geometry.py','visuals.py')]:touch(p,'manually_reviewed_source')
manifest=[]
for name,roles in sorted(accessed.items()):
    p=Path(name);st=p.stat();manifest.append({'path':str(p),'sha256':sha(p),'bytes':st.st_size,'mtime_utc':datetime.fromtimestamp(st.st_mtime,timezone.utc).isoformat(),'roles':sorted(roles)})
write(TRACE/'inputs.json',{'generated_utc':datetime.now(timezone.utc).isoformat(),'run':'horizon_restore_001','files':manifest,'new_run_core_files':sum('new_run_core_manifest' in v['roles'] for v in manifest),'inherited_data_policy':'hash only; new frozen prefix and necessary historical metadata explicitly read; no old raw arrays reanalysed'})
print(json.dumps({'checks':len(checks),'failures':failures,'domains':{k:v for k,v in domains.items() if k in ('budget','dense','execution')}},default=plain,ensure_ascii=False),flush=True)

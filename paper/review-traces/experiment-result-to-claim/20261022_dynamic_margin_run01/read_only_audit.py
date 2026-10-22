"""Fresh bounded review: saved arrays, hashes and algebra only; no project imports.

Never executes simulator, model compilation, optimizer, runner or resume. Only this
trace directory is written. Historical physics is not repeated: hashes and exact
reference/identity arrays establish eligibility for the three already audited cases.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, zipfile, xml.etree.ElementTree as ET
from collections import Counter
import numpy as np
from scipy.interpolate import BSpline
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[4]
TRACE = Path(__file__).resolve().parent
RUN = ROOT/'artifacts/v7_1/dynamic_margin_001'
OLD = ROOT/'artifacts/v7_1/dynamic_reference_001'
seen = set()
def sha(p):
    p=Path(p); seen.add(p); h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
def read(p):
    p=Path(p);seen.add(p);return json.loads(p.read_text(encoding='utf-8-sig'))
def array(p):
    p=Path(p);seen.add(p)
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def mx(x):return float(np.max(np.abs(x)))
def json_default(x):
    if isinstance(x,np.generic):return x.item()
    raise TypeError(type(x).__name__)
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def check_map(mapping):return [p for p,h in mapping.items() if sha(ROOT/p)!=h]
def ratios(x,limit):
    a=np.abs(x)/limit;k,j=np.unravel_index(a.argmax(),a.shape)
    return dict(maximum=float(a[k,j]),index=[int(k),int(j)],value=float(x[k,j]),limit=float(limit[j]),violations=int(np.sum(a>1)))
def extrema(sp):
    out=np.empty((3,2,17));out[:,0]=np.inf;out[:,1]=-np.inf
    joins=np.zeros(3);previous=None
    for left,right in zip(np.unique(sp.t)[:-1],np.unique(sp.t)[1:]):
        w=right-left
        p=np.array([sp((left+right)/2,3)/6,sp(left,2)/2,sp(left,1),sp(left)])
        for order in range(3):
            if previous is not None:
                end=np.array([np.polyval(np.polyder(previous[:,j],order),width) for j in range(17)])
                start=np.array([np.polyval(np.polyder(p[:,j],order),0) for j in range(17)])
                joins[order]=max(joins[order],mx(end-start))
            for j in range(17):
                c=np.polyder(p[:,j],order);roots=np.roots(np.polyder(c))
                x=[0,w]+[float(z.real) for z in roots if abs(z.imag)<1e-10 and 0<z.real<w]
                y=np.polyval(c,x);out[order,0,j]=min(out[order,0,j],min(y));out[order,1,j]=max(out[order,1,j],max(y))
        previous,width=p,w
    return out,joins

run=read(RUN/'run.json');ident=run['identity'];P=ident['protocol']
oldrun=read(OLD/'run.json');oldid=oldrun['identity']
selection=read(ROOT/'ff_prior_dynamic_margin/evidence/selection.json')
binding=read(ROOT/'ff_prior_dynamic_margin/evidence/binding.json')
freeze=read(ROOT/'artifacts/v7_1/waypoint_moving_gate_001/freeze.json')
budget=read(RUN/'budget.json');status=read(RUN/'status.json');summary=read(RUN/'summary.json')
stats=read(ROOT/'ff_prior_dynamic_reference/evidence/completed_statistics.json')
config=read(ROOT/'tracking_diffusion/config/local10mm_v1.json')
urdf_path=ROOT/'dual_arm_space_robot_2026/urdf/dual_arm_space_robot_2026.urdf';seen.add(urdf_path)
urdf=ET.parse(urdf_path).getroot();joints={j.attrib['name']:j for j in urdf.findall('joint')}
for f in ['ff_prior_dynamic_margin/docs/EXPERIMENT_PLAN.md','paper/EXPERIMENT_PLAN.md',
          'ff_prior_dynamic_margin/job.py','ff_prior_dynamic_margin/job.ps1',
          'paper/review-traces/experiment-result-to-claim/20261022_dynamic_reference_run01/read_only_audit.py',
          '.venv/Lib/site-packages/mujoco/include/mujoco/mjdata.h']:
    seen.add(ROOT/f)
for f in ['C:/Users/admin/.codex/skills/experiment-result-to-claim/SKILL.md',
          'C:/Users/admin/.codex/AGENTS.md',
          'C:/Users/admin/.codex/skills/shared-references/review-tracing.md',
          'C:/Users/admin/.codex/skills/shared-references/assurance-contract.md']:
    seen.add(Path(f))
source_errors=check_map(ident['source_hashes']);input_errors=check_map(ident['binding']['files'])
old_source_errors=check_map(oldid['source_hashes'])
seen.add(RUN/'executed_source.zip')
with zipfile.ZipFile(RUN/'executed_source.zip') as z:
    zip_members=set(z.namelist())
    zip_errors=[p for p,h in ident['source_hashes'].items() if p not in zip_members or hashlib.sha256(z.read(p)).hexdigest()!=h]
oldfiles={p.relative_to(ROOT).as_posix() for p in OLD.rglob('*') if p.is_file()}
boundfiles={Path(p).as_posix() for p in ident['binding']['files']}
evidence=dict(review_method='Fresh independent saved-array algebra, frozen-source inspection and exact provenance checks; no simulation/optimizer/resume.',
    producer_head=run['producer_head'],
    identity=dict(source_count=len(ident['source_hashes']),expected_source_count=168,source_errors=source_errors,
        zip_members=len(zip_members),zip_members_exact=zip_members==set(ident['source_hashes']),zip_errors=zip_errors,
        input_count=len(ident['binding']['files']),expected_input_count=151,input_errors=input_errors,
        binding_exact=binding==ident['binding'],selection_exact=selection==ident['selection'],
        original_four_selection_exact=selection['rows']==freeze['certification'],
        old_source_count=len(oldid['source_hashes']),expected_old_source_count=166,old_source_errors=old_source_errors,
        old_file_count=len(oldfiles),old_file_binding_missing=sorted(oldfiles-boundfiles)),
    protocol=dict(rho=P['acceleration_fraction'],rho_from_frozen_residual=1-2*stats['max_observed_response_residual_ratio'],
        seeds=[r['seed'] for r in selection['rows']],splits=[r['split'] for r in selection['rows']],
        original_acceptance_limits_unchanged=True,one_repair_only=True),references=[],reuse=[])

for i,row in enumerate(selection['rows']):
    folder=RUN/f't{i:02}';ref=array(folder/'reference.npz');previous_ref=array(OLD/f't{i:02}'/'reference.npz')
    rep=read(folder/'representation.json');limits=read(folder/'limits.json');caseid=read(folder/'identity.json')
    source=array(ROOT/row['path']/'trace.npz');source_result=read(ROOT/row['path']/'result.json')
    c=ref['coefficients'];t=ref['knots'];n=len(c);sp=BSpline(t,c,3,extrapolate=False)
    limv=np.asarray(limits['velocity']);lima=np.asarray(limits['acceleration'])
    dc=3*np.diff(c,axis=0)/(t[4:n+3]-t[1:n])[:,None]
    ddc=2*np.diff(dc,axis=0)/(t[4:n+2]-t[2:n])[:,None]
    bounds,joins=extrema(sp)
    basis=BSpline(t,np.eye(n),3,extrapolate=False);b=basis(ref['source_time']);v=basis(ref['source_time'],1)
    H=(b.T@b+P['derivative_fit_weight']*v.T@v)/len(b)+P['ridge']*np.eye(n)
    target=(b.T@ref['source_q']+P['derivative_fit_weight']*v.T@ref['source_dq'])/len(b)
    gradient=H@c-target
    objectives=np.sum(c*(.5*(H@c)-target),axis=0)
    unconstrained=[r['joint'] for r in rep['solver'] if r['method']=='UNCONSTRAINED_OPTIMUM_FEASIBLE']
    lower=np.asarray(limits['lower']);upper=np.asarray(limits['upper'])
    refgates=dict(finite=bool(np.isfinite(c).all()),position=bool(np.all(bounds[0,0]>=lower) and np.all(bounds[0,1]<=upper)),
        velocity=bool(np.all(abs(bounds[1])<=limv)),acceleration=bool(np.all(abs(bounds[2])<=lima)),
        initial_q=mx(sp(0)-ref['source_q'][0])<=P['initial_tolerance'],initial_dq=mx(sp(0,1)-ref['source_dq'][0])<=P['initial_tolerance'],
        solver=all(s['success'] for s in rep['solver']))
    rec=dict(case=f't{i:02}',seed=row['seed'],computed_gates=refgates,reported_gates_equal=refgates==rep['gates'],
        original_limits_exact=limits==read(OLD/f't{i:02}'/'limits.json'),
        knots_shape=list(t.shape),coefficient_shape=list(c.shape),
        expected_knots_exact=bool(np.array_equal(t,np.r_[np.zeros(4),np.arange(.1,27-1e-9,.1),np.full(4,27.)])),
        source_arrays_exact={a:bool(np.array_equal(ref[a],source[b])) for a,b in [('source_time','time'),('source_q','q'),('source_dq','dq')]},
        task_exact=caseid['task']==source_result['task'],task_hash_exact=caseid['task_hash']==digest(source_result['task']),
        source_row_exact=caseid['source']==row,waypoints_exact=caseid['waypoints']==source_result['path'],
        source_trace_hash_exact=sha(ROOT/row['path']/'trace.npz')==row['trace_sha256'],
        source_result_hash_exact=sha(ROOT/row['path']/'result.json')==row['result_sha256'],
        c2_join_max_errors=joins.tolist(),polynomial_extrema_report_error=mx(bounds-np.asarray(rep['polynomial_bounds'])),
        analytic_velocity_ratio=float(np.max(abs(bounds[1])/limv)),analytic_acceleration_ratio=float(np.max(abs(bounds[2])/lima)),
        hull_velocity_ratio=float(np.max(abs(dc)/limv)),hull_acceleration_ratio=float(np.max(abs(ddc)/lima)),
        velocity_inward_hull_max_violation=float(np.max(abs(dc)-limv*P['derivative_fraction'])),
        acceleration_margin_hull_max_violation=float(np.max(abs(ddc)-lima*P['acceleration_fraction'])),
        free_coefficient_position_pass=bool(np.all(c[2:]>=lower+P['coefficient_margin']) and np.all(c[2:]<=upper-P['coefficient_margin'])),
        initial_reference_errors=[mx(sp(0)-ref['source_q'][0]),mx(sp(0,1)-ref['source_dq'][0])],
        fit_max_rad=mx(sp(ref['source_time'])-ref['source_q']),fit_rms_rad=float(np.sqrt(np.mean((sp(ref['source_time'])-ref['source_q'])**2))),
        solver_methods=dict(Counter(s['method'] for s in rep['solver'])),solver_iterations=[s['iterations'] for s in rep['solver']],
        objective_report_error=mx(objectives-np.asarray([s['objective'] for s in rep['solver']])),
        unconstrained_stationarity_error=mx(gradient[2:,unconstrained]),
        reference_knots_equal_old=bool(np.array_equal(t,previous_ref['knots'])),
        reference_coefficients_equal_old=bool(np.array_equal(c,previous_ref['coefficients'])),
        changed_coefficient_columns=np.where(np.any(c!=previous_ref['coefficients'],axis=0))[0].tolist())
    evidence['references'].append(rec)
    initial=array(folder/'initial.npz');oldinitial=array(OLD/f't{i:02}'/'initial.npz')
    oldcaseid=read(OLD/f't{i:02}'/'identity.json');report=read(folder/'result.json')
    if i:
        oldreport=read(OLD/f't{i:02}'/'result.json');receipt=read(folder/'reuse.json')
        oldtrace=OLD/f't{i:02}'/'native_trace.npz';oldreplay=OLD/f't{i:02}'/'replay_trace.npz'
        expected=dict(oldreport,new_execution=False,actual_new_steps=0,reused_from=str((OLD/f't{i:02}').relative_to(ROOT)))
        reuse=dict(case=f't{i:02}',reference_exact=rec['reference_knots_equal_old'] and rec['reference_coefficients_equal_old'],
            identity_exact=caseid==oldcaseid,identity_digest_exact=digest(caseid)==digest(oldcaseid),
            complete_initial_keys_exact=set(initial)==set(oldinitial),
            complete_initial_arrays_exact=all(np.array_equal(initial[k],oldinitial[k]) for k in initial),
            limits_exact=limits==read(OLD/f't{i:02}'/'limits.json'),
            source_166_hashes_exact=not old_source_errors,all_old_64_files_bound=not(oldfiles-boundfiles) and not input_errors,
            old_status=oldreport['status'],old_all_gates_pass=all(oldreport['gates'].values()),
            old_native_sha256=sha(oldtrace),old_native_hash_matches_report=sha(oldtrace)==oldreport['native_trace_sha256'],
            old_control_and_replay_evidence_bound=oldtrace.relative_to(ROOT).as_posix() in boundfiles and oldreplay.relative_to(ROOT).as_posix() in boundfiles,
            old_replay_sha256=sha(oldreplay),new_result_is_old_plus_provenance=report==expected,
            receipt=receipt,new_native_trace_absent=not(folder/'native_trace.npz').exists(),new_replay_absent=not(folder/'replay_trace.npz').exists(),
            new_execution=report['new_execution'],actual_new_steps=report['actual_new_steps'])
        reuse['qualified']=all(reuse[k] for k in ['reference_exact','identity_exact','identity_digest_exact','complete_initial_keys_exact',
            'complete_initial_arrays_exact','limits_exact','source_166_hashes_exact','all_old_64_files_bound','old_all_gates_pass',
            'old_native_hash_matches_report','old_control_and_replay_evidence_bound','new_result_is_old_plus_provenance']) and reuse['old_status']=='EXECUTION_VERIFIED' and not reuse['new_execution'] and reuse['actual_new_steps']==0
        evidence['reuse'].append(reuse)

folder=RUN/'t00';identity=read(folder/'identity.json');limits=read(folder/'limits.json');report=read(folder/'result.json')
ref=array(folder/'reference.npz');native=array(folder/'native_trace.npz');replay=array(folder/'replay_trace.npz')
initial=array(folder/'initial.npz')['values'];geo=array(folder/'geometry.npz');pairs=read(folder/'geometry_pairs.json')
row=selection['rows'][0];source=array(ROOT/row['path']/'trace.npz');src_result=read(ROOT/row['path']/'result.json')
B=np.asarray(limits['B']);Bp=np.asarray(limits['Bplus']);vel=np.asarray(limits['velocity']);acc=np.asarray(limits['acceleration'])
original_map=np.zeros((67,17));pattern=np.array([[0,1],[1,0],[1,0],[0,1],[0,1],[1,0],[1,0],[0,1],[0,1],[1,0],[1,0],[0,1]])/6
original_map[:60,:10]=np.kron(np.eye(5),pattern);original_map[60:,10:]=np.eye(7)
evidence['original_physical_contract']=dict(mapping_exact=bool(np.array_equal(B,original_map)),
    inverse_mapping_error=mx(Bp-np.linalg.pinv(original_map)),
    velocity_exact=bool(np.array_equal(vel,np.r_[np.full(10,.8),np.full(7,1.2)])),
    acceleration_exact=bool(np.array_equal(acc,np.r_[np.full(10,2.5),np.full(7,4.)])),
    torque_exact=bool(np.array_equal(limits['native_torque'],np.r_[np.full(60,4.),np.full(7,80.)])),
    lower_exact=bool(np.array_equal(limits['lower'],np.r_[np.full(10,-1.),np.full(7,-np.pi)])),
    upper_exact=bool(np.array_equal(limits['upper'],np.r_[np.full(10,1.),np.full(7,np.pi)])))
mapping=sorted(identity['robot']['joint_index_map'].items(),key=lambda x:x[1]['actuator'])
qids=[x['qpos'] for _,x in mapping];vids=[x['dof'] for _,x in mapping]
q=native['qpos'][:,qids];v=native['qvel'][:,vids];pv=v@Bp.T;a=np.diff(pv,axis=0)/.002;na=np.diff(v,axis=0)/.002
sp=BSpline(ref['knots'],ref['coefficients'],3,extrapolate=False);times=np.arange(1,13501)*.002
position=np.linalg.norm(native['ee'][...,:3,3]-native['desired'][...,:3,3],axis=-1)
rotation=Rotation.from_matrix((np.swapaxes(native['desired'][...,:3,:3],-1,-2)@native['ee'][...,:3,:3]).reshape(-1,3,3)).magnitude().reshape(position.shape)
hold=native['time']>=25.5-1e-9
mp=np.array([np.max(np.linalg.norm(native['momentum'][:,s],axis=1)) for s in [slice(0,3),slice(3,6)]])
me=np.array([np.max(np.linalg.norm(native['exchange'][:,s],axis=1)) for s in [slice(0,3),slice(3,6)]])
limited=[i for i,(name,_) in enumerate(mapping) if joints[name].attrib['type']=='revolute']
lo=np.array([float(joints[mapping[i][0]].find('limit').attrib['lower']) for i in limited])
hi=np.array([float(joints[mapping[i][0]].find('limit').attrib['upper']) for i in limited])
mid=(native['qpos'][:-1]+native['qpos'][1:])/2
for k in (3,77):
    qa=native['qpos'][:-1,k:k+4];qb=native['qpos'][1:,k:k+4]
    qb=qb*np.where(np.sum(qa*qb,axis=1)<0,-1,1)[:,None];qm=qa+qb
    mid[:,k:k+4]=qm/np.linalg.norm(qm,axis=1)[:,None]
path=identity['waypoints'];points=np.array(path['waypoint_points_m']);dur=np.array(path['segment_durations_s']);cum=np.r_[0,np.cumsum(dur)]
desired=[]
for t in native['time']:
    if t<4.5:left=np.array(path['initial_position_w']);right=points[0];s=t/4.5
    elif t>=25.5:left=right=points[-1];s=0
    else:
        seg=min(int(np.searchsorted(cum,t-4.5,side='right')-1),5);left,right=points[seg:seg+2];s=(t-4.5-cum[seg])/dur[seg]
    desired.append(left+s**3*(10-15*s+6*s*s)*(right-left))
target_R=Rotation.from_quat(native['qpos'][:,[78,79,80,77]]).as_matrix();home_R=native['ee'][0,1,:3,:3];home_p=native['ee'][0,1,:3,3]
point=native['qpos'][:,74:77]+target_R@np.array([0,-.205,0]);s=np.clip(native['time']/4.5,0,1);alpha=s**3*(10-15*s+6*s*s)
rigid_p=home_p+alpha[:,None]*(point-home_p)+(64*s**3*(1-s)**3)[:,None]*np.array(config['approach_offset'])
target_goal_R=target_R@(target_R[0].T@home_R);rv=Rotation.from_matrix(target_goal_R@home_R.T).as_rotvec()
rigid_R=Rotation.from_rotvec(alpha[:,None]*rv).as_matrix()@home_R
transitions=[]
for boundary in [4.5,25.5]:
    tr=array(folder/f'transition_{int(boundary*10)}.npz');k=int(round(boundary/.002))
    transitions.append(dict(boundary=boundary,index=k,times=tr['time'].tolist(),
        saved_slice_errors={name:mx(tr[name]-native[name][k-1:k+2]) for name in tr},
        full_replay_slice_error=mx(tr['integration']-replay['integration'][k-1:k+2]),
        state_step_max=mx(np.diff(tr['qpos'],axis=0)),velocity_step_max=mx(np.diff(tr['qvel'],axis=0))))
replay_errors={k:mx(native[k]-replay[k]) for k in ['qpos','qvel','ctrl','integration']}
gates=dict(complete_native_time=len(native['ctrl'])==13500 and report['stop_reason'] is None and abs(native['time'][-1]-27)<1e-8,
    independent_replay=max(replay_errors.values())<=1e-10,whole_body_clearance=bool(np.min(geo['minima'])>=0),
    geometry_step=report['maximum_geometry_step_m']<=.002+1e-12,
    dynamic_position=bool(np.all(position<=.005)),dynamic_rotation=bool(np.all(rotation<=np.pi/180)),
    relative_hold=bool(hold.any() and np.all(position[hold]<=.005) and np.all(rotation[hold]<=np.pi/180)),
    joint_limits=bool(np.all(native['q17']>=limits['lower']) and np.all(native['q17']<=limits['upper']) and np.all(native['qref']>=limits['lower']) and np.all(native['qref']<=limits['upper']) and np.all(q[:,limited]>=lo) and np.all(q[:,limited]<=hi)),
    derivative_limits=bool(np.all(abs(pv)<=vel) and np.all(abs(a)<=acc) and np.all(abs(v)<=abs(B)@vel) and np.all(abs(na)<=abs(B)@acc) and np.all(abs(native['reference_qd'])<=vel) and np.all(abs(native['reference_qdd'])<=acc)),
    raw_torque=bool(np.all(abs(native['raw_torque'])<=np.array(limits['native_torque'])+1e-12)),tracking=mx(native['q17']-native['qref'])<=.02,
    manifold=mx(q-native['q17']@B.T)<=.02,momentum=bool(np.all(mp<=1e-5+.002*me)),
    no_hidden_reset=max(replay_errors.values())==0 and len(native['integration'])==13501,
    no_external_wrench=mx(native['integration'][:,307:])==0,
    finite=all(np.isfinite(x).all() for x in native.values()))
evidence['t00']=dict(computed_gates=gates,reported_gates_equal=gates==report['gates'],
    native_shapes={k:list(x.shape) for k,x in native.items()},replay_shapes={k:list(x.shape) for k,x in replay.items()},
    native_steps=len(native['ctrl']),replay_steps=len(replay['ctrl']),native_trace_sha256=sha(folder/'native_trace.npz'),
    native_hash_matches_report=sha(folder/'native_trace.npz')==report['native_trace_sha256'],
    initial_shape=list(initial.shape),initial_state_id_exact=identity['initial_state_id']==digest(dict(values=initial.tolist(),model_hash=identity['robot']['scene_model_hash'],state_mask=8191)),
    initial_native_error=mx(initial-native['integration'][0]),initial_replay_error=mx(initial-replay['integration'][0]),
    old_full_identity_exact=identity==read(OLD/'t00/identity.json'),old_initial_array_exact=bool(np.array_equal(initial,array(OLD/'t00/initial.npz')['values'])),
    source_initial_qpos_error=mx(native['qpos'][0]-source['qpos'][0]),source_initial_qvel_error=mx(native['qvel'][0]-source['qvel'][0]),
    integration_components=dict(time=mx(native['integration'][:,0]-native['time']),qpos=mx(native['integration'][:,1:82]-native['qpos']),qvel=mx(native['integration'][:,82:161]-native['qvel']),ctrl=mx(native['integration'][1:,240:307]-native['ctrl'])),
    external_force_max=mx(native['integration'][:,307:386]),external_wrench_max=mx(native['integration'][:,386:836]),
    target_initial_twist_error=mx(native['qvel'][0,-6:]-np.r_[src_result['task_parameters']['target_velocity'],src_result['task_parameters']['target_omega_local']]),
    time_start=float(native['time'][0]),time_end=float(native['time'][-1]),time_step_error=mx(np.diff(native['time'])-.002),
    reference_command_errors=[mx(native['qref'][1:]-sp(times)),mx(native['reference_qd']-sp(times,1)),mx(native['reference_qdd']-sp(times,2))],
    actual_planner_velocity=ratios(pv,vel),actual_planner_acceleration=ratios(a,acc),actual_native_velocity=ratios(v,abs(B)@vel),actual_native_acceleration=ratios(na,abs(B)@acc),
    position_max_mm=(np.max(position,axis=0)*1000).tolist(),position_final_mm=(position[-1]*1000).tolist(),rotation_max_deg=np.rad2deg(np.max(rotation,axis=0)).tolist(),
    hold_position_max_mm=(np.max(position[hold],axis=0)*1000).tolist(),hold_rotation_max_deg=np.rad2deg(np.max(rotation[hold],axis=0)).tolist(),
    desired_continuum_position_error=mx(np.array(desired)-native['desired'][:,0,:3,3]),
    desired_continuum_orientation_error=mx(native['desired'][:,0,:3,:3]-np.asarray(path['target_rotation_world'])),
    desired_rigid_position_error=mx(rigid_p-native['desired'][:,1,:3,3]),desired_rigid_orientation_error=mx(rigid_R-native['desired'][:,1,:3,:3]),
    torque_max_ratio=float(np.max(abs(native['raw_torque'])/limits['native_torque'])),raw_control_error=mx(native['raw_torque']-native['ctrl']),
    tracking_max_rad=mx(native['q17']-native['qref']),manifold_max_rad=mx(q-native['q17']@B.T),planner_mapping_error=mx(native['q17']-q@Bp.T),
    native_limited_joint_count=len(limited),momentum_peak=mp.tolist(),momentum_threshold=(1e-5+.002*me).tolist(),
    replay_errors=replay_errors,transitions=transitions,
    geometry=dict(scan_count=len(geo['qpos']),witness_count=len(pairs),
        initial_error=mx(geo['qpos'][0]-native['qpos'][0]),all_endpoint_errors=mx(geo['qpos'][1::2]-native['qpos'][1:]),all_midpoint_errors=mx(geo['qpos'][2::2]-mid),
        clearance_reduction_error=mx(native['clearance'][1:]-np.minimum(geo['minima'][1::2],geo['minima'][2::2])),
        minimum_clearance_mm=float(np.min(geo['minima'])*1000),reported_geometry_step_m=report['maximum_geometry_step_m'],
        unique_pair_witnesses=list({json.dumps(p,sort_keys=True):p for p in pairs}.values()),
        distance_scope='Saved all-pair-policy minima with conservative pruning and every midpoint coverage checked. No collision engine rerun; no continuous-time certificate.'))

# Independent URDF transforms on every saved endpoint AND midpoint. Chronological
# ordering avoids assuming a joint-space midpoint is a rigid-body pose midpoint.
# This is a sampled-displacement bound, not a continuous-time swept certificate.
algebra_qpos=np.empty_like(geo['qpos']);algebra_qpos[0]=geo['qpos'][0]
algebra_qpos[1::2]=geo['qpos'][2::2];algebra_qpos[2::2]=geo['qpos'][1::2]
N=len(algebra_qpos);frames={'world':(np.zeros((N,3)),np.broadcast_to(np.eye(3),(N,3,3)))}
pending=list(urdf.findall('joint'))
while pending:
    progress=False
    for joint in pending[:]:
        parent=joint.find('parent').attrib['link'];child=joint.find('child').attrib['link']
        if parent not in frames:continue
        typ=joint.attrib['type'];name=joint.attrib['name'];pp,pr=frames[parent]
        if typ=='floating':
            offset=0 if name=='world_joint' else 74
            pos=algebra_qpos[:,offset:offset+3];quat=algebra_qpos[:,offset+3:offset+7]
            rot=Rotation.from_quat(quat[:,[1,2,3,0]]).as_matrix()
        else:
            origin=joint.find('origin');xyz=np.fromstring(origin.attrib.get('xyz','0 0 0') if origin is not None else '0 0 0',sep=' ')
            rpy=np.fromstring(origin.attrib.get('rpy','0 0 0') if origin is not None else '0 0 0',sep=' ')
            pos=pp+pr@xyz;rot=pr@Rotation.from_euler('xyz',rpy).as_matrix()
            if typ in ['revolute','continuous']:
                axis=np.fromstring(joint.find('axis').attrib.get('xyz','1 0 0'),sep=' ');axis=axis/np.linalg.norm(axis)
                angle=algebra_qpos[:,identity['robot']['joint_index_map'][name]['qpos']]
                rot=rot@Rotation.from_rotvec(angle[:,None]*axis).as_matrix()
        frames[child]=(pos,rot);pending.remove(joint);progress=True
    if not progress:raise ValueError('unresolved URDF tree')
fk_errors=[]
for i,name in enumerate(['link_30','end_effector_r']):
    pos,rot=frames[name]
    if i==0:pos=pos+rot@np.array([.0475,0,0])
    fk_errors.append(dict(body=name,position=mx(pos[::2]-native['ee'][:,i,:3,3]),rotation=mx(rot[::2]-native['ee'][:,i,:3,:3])))
mesh_radii={};largest=0.;largest_name='';geom_count=0
for link in urdf.findall('link'):
    body=link.attrib['name'];bp,br=frames[body]
    angles=Rotation.from_matrix(br[1:]@br[:-1].transpose(0,2,1)).magnitude()
    for geom in list(link.findall('collision'))+list(link.findall('visual')):
        origin=geom.find('origin');xyz=np.fromstring(origin.attrib.get('xyz','0 0 0') if origin is not None else '0 0 0',sep=' ')
        shape=list(geom.find('geometry'))[0]
        if shape.tag=='mesh':
            filename=shape.attrib['filename'];scale=np.fromstring(shape.attrib.get('scale','1 1 1'),sep=' ')
            if filename not in mesh_radii:
                mesh=ROOT/'dual_arm_space_robot_2026/meshes'/filename;seen.add(mesh)
                with mesh.open('rb') as f:
                    header=f.read(84);count=int.from_bytes(header[80:84],'little')
                    dtype=np.dtype([('normal','<f4',(3,)),('vertices','<f4',(3,3)),('attr','<u2')])
                    verts=np.fromfile(f,dtype=dtype,count=count)['vertices']
                mesh_radii[filename]=float(np.sqrt(np.max(np.sum((verts.astype(float)*scale)**2,axis=-1))))
            radius=mesh_radii[filename]
        elif shape.tag=='box':radius=float(np.linalg.norm(np.fromstring(shape.attrib['size'],sep=' ')/2))
        else:raise ValueError('Unexpected geometry')
        centers=bp+br@xyz
        # For any point in the original mesh, ||delta center|| + angle*radius bounds motion.
        motion=np.linalg.norm(np.diff(centers,axis=0),axis=1)+angles*radius
        halfbound=float(np.max(motion));geom_count+=1
        if halfbound>largest:largest,largest_name=halfbound,geom.attrib.get('name',body)
evidence['t00']['geometry']['independent_URDF_FK_errors']=fk_errors
evidence['t00']['geometry']['independent_mesh_origin_sample_step_bound_m']=largest
evidence['t00']['geometry']['bound_largest_geom']=largest_name
evidence['t00']['geometry']['URDF_visual_and_collision_geometries_checked']=geom_count
evidence['t00']['geometry']['conservative_bound_pass']=largest<=.002
evidence['t00']['geometry']['bound_scope']='All URDF visual/collision meshes and boxes at every native and midpoint state, including target; static scene obstacles have zero motion. Consecutive saved-state displacement bound, not a continuous-time certificate; need not equal MuJoCo centered-rbound metric.'

events_path=RUN/'events.jsonl';seen.add(events_path)
events=[json.loads(line) for line in events_path.read_text().splitlines()]
stages=Counter(e['stage'] for e in events)
oldsummary=read(OLD/'summary.json')
evidence['budget']=dict(status=status['status'],monitor_elapsed_s=status['elapsed_s'],experiment_elapsed_s=budget['elapsed_s'],
    native_steps_actual=len(native['ctrl']),replay_steps_actual=len(replay['ctrl']),new_step_total=len(native['ctrl'])+len(replay['ctrl']),
    budget_matches_arrays=budget['native_steps_reserved']==len(native['ctrl']) and budget['replay_steps_reserved']==len(replay['ctrl']),
    new_executions=1,historical_reuses=sum(r['qualified'] for r in evidence['reuse']),logical_units=4,
    recorded_output_bytes=budget['output_bytes'],actual_run_output_bytes=sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file()),
    limits=dict(threads=2,wall_s=600,bytes=1024**3,new_steps=27000),
    event_stages=dict(stages),native_event_cases=sorted({e['case'] for e in events if e['stage']=='native_torque'}),
    old_run_denominator=oldsummary['denominator'],old_run_verified=oldsummary['verified'],
    cumulative_real_execution_attempts=5,cumulative_real_verified_executions=4,cumulative_failed_execution_attempts=1,
    original_P2_total_steps=432000,old_P2a_spent_steps=108000,repair_spent_steps=27000,P2b_remaining_new_steps=297000,
    P2b_original_logical_units=12,P2b_maximum_new_units=11,P2b_required_exact_reused_units_minimum=1,
    P2b_reuse_eligibility_not_established_by_this_review=True,training_updates=0,TRAIN_added=0,DEV_executions=0,TEST=False)
evidence['paper_claim_audit']='unavailable' if not (ROOT/'paper/PAPER_CLAIM_AUDIT.json').exists() else read(ROOT/'paper/PAPER_CLAIM_AUDIT.json')
evidence['overall']=dict(local_P2a_gate=all(gates.values()) and all(all(r['computed_gates'].values()) for r in evidence['references']) and all(r['qualified'] for r in evidence['reuse']) and not(source_errors or input_errors or zip_errors or old_source_errors) and largest<=.002,
    claim_supported='partial',C1='Only P2a four fixed TRAIN reference/execution qualifications; complete task-input system and unseen/generalization claims remain unsupported.',
    C2='Not evaluated: no paired Diffusion comparison, training or DEV/TEST.',
    paper_verdict='provisional — no paper claim audit available',continuous_time_certificate=False,hardware=False)
stamp=datetime.now(timezone.utc).isoformat()
(TRACE/'saved_array_audit.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2,allow_nan=False,default=json_default),encoding='utf-8')
seen.add(Path(__file__).resolve())
manifest=[]
for p in sorted(seen):
    manifest.append(dict(path=str(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p).replace('\\','/'),sha256=sha(p),bytes=p.stat().st_size,mtime_utc=datetime.fromtimestamp(p.stat().st_mtime,timezone.utc).isoformat()))
(TRACE/'inputs.json').write_text(json.dumps(dict(reviewed_at_utc=stamp,method=evidence['review_method'],files=manifest),ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(identity=evidence['identity'],references=evidence['references'],reuse=evidence['reuse'],t00=evidence['t00'],budget=evidence['budget'],overall=evidence['overall']),ensure_ascii=True,indent=2,default=json_default))

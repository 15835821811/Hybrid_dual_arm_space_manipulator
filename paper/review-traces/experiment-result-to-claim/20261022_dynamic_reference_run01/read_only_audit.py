"""Independent saved-array audit only; no project imports, optimizer, or physics calls."""
from pathlib import Path
import hashlib
import json
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
import numpy as np
from scipy.interpolate import BSpline
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[4]
TRACE = Path(__file__).resolve().parent
RUN = ROOT / 'artifacts/v7_1/dynamic_reference_001'
seen = set()
seen.update([ROOT/'paper/EXPERIMENT_PLAN.md',
    Path('C:/Users/admin/.codex/skills/experiment-result-to-claim/SKILL.md'),
    Path('C:/Users/admin/.codex/skills/shared-references/review-tracing.md'),
    ROOT/'.venv/Lib/site-packages/mujoco/include/mujoco/mjdata.h'])

def read(path):
    path = Path(path); seen.add(path)
    return json.loads(path.read_text(encoding='utf-8-sig'))

def arrays(path):
    path = Path(path); seen.add(path)
    with np.load(path, allow_pickle=False) as z:
        return {k:z[k] for k in z.files}

def hash_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576), b''): h.update(block)
    return h.hexdigest()

def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def maximum(x):
    return float(np.max(np.abs(x)))

def ratios(arr, limit):
    ratio = np.abs(arr)/limit
    k,j = np.unravel_index(np.argmax(ratio), ratio.shape)
    bad = np.argwhere(ratio > 1)
    return dict(maximum=float(ratio[k,j]), index=[int(k),int(j)], value=float(arr[k,j]),
                limit=float(np.asarray(limit)[j]), violations=bad.tolist())

def extrema(spline):
    """Taylor coefficients on each of 270 positive knot spans; roots of derivatives."""
    breaks = np.unique(spline.t)
    bounds = np.empty((3,2,17)); bounds[:,0] = np.inf; bounds[:,1] = -np.inf
    continuity = np.zeros(3)
    previous = None
    for left,right in zip(breaks[:-1],breaks[1:]):
        width = right-left
        # The cubic third derivative at the span midpoint is unambiguous.
        poly = np.array([spline((left+right)/2,3)/6, spline(left,2)/2,
                         spline(left,1), spline(left)])
        if previous is not None:
            for order in range(3):
                prev = np.array([np.polyval(np.polyder(previous[:,j],order),prev_width) for j in range(17)])
                cur = np.array([np.polyval(np.polyder(poly[:,j],order),0) for j in range(17)])
                continuity[order] = max(continuity[order],maximum(prev-cur))
        for order in range(3):
            for j in range(17):
                c = np.polyder(poly[:,j],order)
                roots = np.roots(np.polyder(c))
                xx = [0.,width]+[float(r.real) for r in roots if abs(r.imag)<1e-10 and 0<r.real<width]
                yy = np.polyval(c,xx)
                bounds[order,0,j] = min(bounds[order,0,j],float(min(yy)))
                bounds[order,1,j] = max(bounds[order,1,j],float(max(yy)))
        previous,prev_width = poly,width
    return bounds,continuity

run = read(RUN/'run.json'); ident = run['identity']
frozen = read(ROOT/'artifacts/v7_1/waypoint_moving_gate_001/freeze.json')
selection = read(ROOT/'ff_prior_dynamic_reference/evidence/selection.json')
binding = read(ROOT/'ff_prior_dynamic_reference/evidence/binding.json')
budget = read(RUN/'budget.json'); config=read(ROOT/'tracking_diffusion/config/local10mm_v1.json')
urdf_path=ROOT/'dual_arm_space_robot_2026/urdf/dual_arm_space_robot_2026.urdf'
seen.add(urdf_path); urdf=ET.parse(urdf_path).getroot()
urdf_joints={j.attrib['name']:j for j in urdf.findall('joint')}
seen.add(RUN/'executed_source.zip')
source_errors=[]; binding_errors=[]
with zipfile.ZipFile(RUN/'executed_source.zip') as z:
    members=set(z.namelist())
    for path,h in ident['source_hashes'].items():
        f=ROOT/path; seen.add(f)
        if path not in members or hashlib.sha256(z.read(path)).hexdigest()!=h or hash_file(f)!=h:
            source_errors.append(path)
for path,h in ident['binding']['files'].items():
    f=ROOT/path;seen.add(f)
    if hash_file(f)!=h:binding_errors.append(path)
evidence=dict(scope='Only this completed run and its frozen inputs; saved-array algebra and source inspection.',
    no_physics_or_optimizer_invoked=True, producer_head=run['producer_head'],
    source_identity=dict(expected=166,actual=len(ident['source_hashes']),zip_members=len(members),
                         same_members=members==set(ident['source_hashes']),errors=source_errors),
    input_identity=dict(expected=86,actual=len(ident['binding']['files']),errors=binding_errors,
                        binding_file_equal=binding==ident['binding']),
    selection=dict(run_equal_freeze=ident['selection']['rows']==frozen['certification'],
                   selection_file_equal=ident['selection']==selection,
                   seeds=[r['seed'] for r in selection['rows']],splits=[r['split'] for r in selection['rows']]),
    cases=[])
native_total=replay_total=0
for i,row in enumerate(selection['rows']):
    directory=RUN/f't{i:02}'
    identity=read(directory/'identity.json'); report=read(directory/'result.json')
    representation=read(directory/'representation.json'); limits=read(directory/'limits.json')
    old=read(ROOT/row['path']/'result.json'); source=arrays(ROOT/row['path']/'trace.npz')
    ref=arrays(directory/'reference.npz'); native=arrays(directory/'native_trace.npz')
    replay=arrays(directory/'replay_trace.npz'); initial=arrays(directory/'initial.npz')['values']
    geometry=arrays(directory/'geometry.npz'); pairs=read(directory/'geometry_pairs.json')
    B=np.asarray(limits['B']); Bp=np.asarray(limits['Bplus'])
    mapping=sorted(identity['robot']['joint_index_map'].items(),key=lambda kv:kv[1]['actuator'])
    qids=[j['qpos'] for _,j in mapping]; vids=[j['dof'] for _,j in mapping]
    q=native['qpos'][:,qids]; v=native['qvel'][:,vids]
    limited_indices=[k for k,(name,_) in enumerate(mapping) if urdf_joints[name].attrib['type']=='revolute']
    native_lower=np.array([float(urdf_joints[mapping[k][0]].find('limit').attrib['lower']) for k in limited_indices])
    native_upper=np.array([float(urdf_joints[mapping[k][0]].find('limit').attrib['upper']) for k in limited_indices])
    planner_v=v@Bp.T; a=np.diff(planner_v,axis=0)/.002; lowa=np.diff(v,axis=0)/.002
    vel=np.asarray(limits['velocity']); acc=np.asarray(limits['acceleration'])
    sp=BSpline(ref['knots'],ref['coefficients'],3,extrapolate=False)
    bounds,c2=extrema(sp)
    ncoef=len(ref['coefficients']); knots=ref['knots']
    dc=3*np.diff(ref['coefficients'],axis=0)/(knots[4:ncoef+3]-knots[1:ncoef])[:,None]
    ddc=2*np.diff(dc,axis=0)/(knots[4:ncoef+2]-knots[2:ncoef])[:,None]
    time=np.arange(1,13501)*.002
    derivative_map_error=[maximum(BSpline(knots[1:-1],dc,2)(time)-sp(time,1)),
                          maximum(BSpline(knots[2:-2],ddc,1)(time)-sp(time,2))]
    ff=native['reference_qdd']@B.T
    w=np.r_[np.full(60,42.),np.full(7,34.)]
    pp=(native['qref'][1:]@B.T-q[:-1])*w**2
    dd=(native['reference_qd']@B.T-v[:-1])*2*w
    requested=ff+pp+dd
    position=np.linalg.norm(native['ee'][...,:3,3]-native['desired'][...,:3,3],axis=-1)
    rotmat=np.swapaxes(native['desired'][...,:3,:3],-1,-2)@native['ee'][...,:3,:3]
    rotation=Rotation.from_matrix(rotmat.reshape(-1,3,3)).magnitude().reshape(position.shape)
    # Check every geometry scan is an actual state or its midpoint, with quaternion geodesic midpoint.
    midpoint=(native['qpos'][:-1]+native['qpos'][1:])/2
    for start in (3,77):
        qa=native['qpos'][:-1,start:start+4]; qb=native['qpos'][1:,start:start+4]
        qb=qb*np.where(np.sum(qa*qb,axis=1)<0,-1,1)[:,None]
        qm=qa+qb; midpoint[:,start:start+4]=qm/np.linalg.norm(qm,axis=1)[:,None]
    coverage=dict(scan_count=len(geometry['qpos']),pair_witness_count=len(pairs),
        initial_error=maximum(geometry['qpos'][0]-native['qpos'][0]),
        all_native_endpoint_error=maximum(geometry['qpos'][1::2]-native['qpos'][1:]),
        all_midpoint_error=maximum(geometry['qpos'][2::2]-midpoint),
        clearance_aggregation_error=maximum(native['clearance'][1:]-np.minimum(geometry['minima'][1::2],geometry['minima'][2::2])),
        minimum_clearance_mm=float(np.min(geometry['minima'])*1000),
        maximum_geometry_step_m_reported=report['maximum_geometry_step_m'],
        geometry_step_not_recomputed_without_geometry_transforms=True)
    # Continuum desired world path, independently reconstructed from frozen waypoint metadata.
    path=identity['waypoints']; points=np.asarray(path['waypoint_points_m']); durations=np.asarray(path['segment_durations_s'])
    cum=np.r_[0,np.cumsum(durations)]; path_positions=[]
    for t in native['time']:
        if t<4.5: left=np.asarray(path['initial_position_w']);right=points[0];s=t/4.5
        elif t>=25.5:left=right=points[-1];s=0.
        else:
            segment=min(int(np.searchsorted(cum,t-4.5,side='right')-1),5)
            left,right=points[segment:segment+2];s=(t-4.5-cum[segment])/durations[segment]
        path_positions.append(left+s**3*(10-15*s+6*s*s)*(right-left))
    # Rigid reference reconstructed from saved target state, original grasp and approach contracts.
    target_R=Rotation.from_quat(native['qpos'][:,[78,79,80,77]]).as_matrix()
    home_R=native['ee'][0,1,:3,:3]; home_p=native['ee'][0,1,:3,3]
    local_R=target_R[0].T@home_R
    point=native['qpos'][:,74:77]+target_R@np.array([0.,-.205,0.])
    s=np.clip(native['time']/4.5,0,1);alpha=s**3*(10-15*s+6*s*s)
    rigid_p=home_p+alpha[:,None]*(point-home_p)+(64*s**3*(1-s)**3)[:,None]*np.array(config['approach_offset'])
    rr=target_R@local_R
    rv=Rotation.from_matrix(rr@home_R.T).as_rotvec()
    rigid_R=Rotation.from_rotvec(alpha[:,None]*rv).as_matrix()@home_R
    transition=[]
    for boundary in [4.5,25.5]:
        tr=arrays(directory/f'transition_{int(boundary*10)}.npz');k=int(round(boundary/.002))
        transition.append(dict(boundary=boundary,index=k,errors={name:maximum(tr[name]-native[name][k-1:k+2]) for name in tr},times=tr['time'].tolist()))
    mp=np.array([np.max(np.linalg.norm(native['momentum'][:,sl],axis=1)) for sl in (slice(0,3),slice(3,6))])
    me=np.array([np.max(np.linalg.norm(native['exchange'][:,sl],axis=1)) for sl in (slice(0,3),slice(3,6))])
    case=dict(case=f't{i:02}',seed=row['seed'],status_reported=report['status'],
        source_arrays_equal={a:bool(np.array_equal(ref[a],source[b])) for a,b in [('source_time','time'),('source_q','q'),('source_dq','dq')]},
        task_equal=identity['task']==old['task'],task_hash_correct=identity['task_hash']==digest(old['task']),
        waypoint_metadata_equal=path==old['path'],waypoint_count=path['waypoint_count'],duration=identity['task']['duration'],
        scene_equal_profile=all(o['object_id']==r['name'] and o['position']==r['center'] and o['size'][0]==r['radius'] for o,r in zip(identity['scene']['objects'],config['obstacles'])),
        full_initial_qpos_error=maximum(native['qpos'][0]-source['qpos'][0]),
        full_initial_qvel_error=maximum(native['qvel'][0]-source['qvel'][0]),
        initial_integration_shape=list(initial.shape),initial_integration_error=maximum(initial-native['integration'][0]),
        initial_replay_error=maximum(initial-replay['integration'][0]),
        initial_state_id_correct=identity['initial_state_id']==digest(dict(values=initial.tolist(),model_hash=identity['robot']['scene_model_hash'],state_mask=8191)),
        integration_component_errors=dict(time=maximum(native['integration'][:,0]-native['time']),qpos=maximum(native['integration'][:,1:82]-native['qpos']),qvel=maximum(native['integration'][:,82:161]-native['qvel']),ctrl=maximum(native['integration'][1:,240:307]-native['ctrl'])),
        external_applied_force_max=maximum(native['integration'][:,307:386]),external_applied_wrench_max=maximum(native['integration'][:,386:836]),
        target_initial_velocity_error=maximum(native['qvel'][0,-6:]-np.r_[old['task_parameters']['target_velocity'],old['task_parameters']['target_omega_local']]),
        basis_shape=list(ref['coefficients'].shape),derivative_mapping_error=derivative_map_error,
        c2_join_errors=c2.tolist(),exact_extrema_difference=maximum(bounds-np.asarray(representation['polynomial_bounds'])),
        analytic_velocity_ratio=float(np.max(np.abs(bounds[1])/vel)),analytic_acceleration_ratio=float(np.max(np.abs(bounds[2])/acc)),
        analytic_position_pass=bool(np.all(bounds[0,0]>=limits['lower']) and np.all(bounds[0,1]<=limits['upper'])),
        hull_velocity_ratio=float(np.max(abs(dc)/vel)),hull_acceleration_ratio=float(np.max(abs(ddc)/acc)),
        hull_violation_to_inward_constraints=float(max(np.max(abs(dc)-vel*(1-1e-8)),np.max(abs(ddc)-acc*(1-1e-8)))),
        free_coefficient_position_pass=bool(np.all(ref['coefficients'][2:]>=np.asarray(limits['lower'])+1e-9) and np.all(ref['coefficients'][2:]<=np.asarray(limits['upper'])-1e-9)),
        initial_reference_position_error=maximum(sp(0)-ref['source_q'][0]),initial_reference_velocity_error=maximum(sp(0,1)-ref['source_dq'][0]),
        fit_max_rad=maximum(sp(ref['source_time'])-ref['source_q']),fit_rms_rad=float(np.sqrt(np.mean((sp(ref['source_time'])-ref['source_q'])**2))),
        saved_command_errors=[maximum(native['qref'][1:]-sp(time)),maximum(native['reference_qd']-sp(time,1)),maximum(native['reference_qdd']-sp(time,2))],
        actual_velocity=ratios(planner_v,vel),actual_acceleration=ratios(a,acc),
        native_velocity=ratios(v,abs(B)@vel),native_acceleration=ratios(lowa,abs(B)@acc),
        max_actual_reference_acceleration_residual_ratio=float(np.max(abs(a-native['reference_qdd'])/acc)),
        max_native_reference_acceleration_residual_ratio=float(np.max(abs(lowa-ff)/(abs(B)@acc))),
        requested_acceleration_max_ratio=float(np.max(abs(requested)/(abs(B)@acc))),
        actual_time_start=float(native['time'][0]),actual_time_end=float(native['time'][-1]),time_step_max_error=maximum(np.diff(native['time'])-.002),
        native_steps=len(native['ctrl']),replay_steps=len(replay['ctrl']),native_state_count=len(native['qpos']),
        native_trace_hash_correct=hash_file(directory/'native_trace.npz')==report['native_trace_sha256'],
        replay_errors={k:maximum(native[k]-replay[k]) for k in ['qpos','qvel','ctrl','integration']},
        world_position_max_mm=(np.max(position,axis=0)*1000).tolist(),world_rotation_max_deg=np.rad2deg(np.max(rotation,axis=0)).tolist(),
        desired_path_position_error=maximum(np.asarray(path_positions)-native['desired'][:,0,:3,3]),
        desired_rigid_position_error=maximum(rigid_p-native['desired'][:,1,:3,3]),desired_rigid_rotation_matrix_error=maximum(rigid_R-native['desired'][:,1,:3,:3]),
        raw_torque_max_ratio=float(np.max(abs(native['raw_torque'])/limits['native_torque'])),raw_control_error=maximum(native['raw_torque']-native['ctrl']),
        actual_domain_pass=bool(np.all(native['q17']>=limits['lower']) and np.all(native['q17']<=limits['upper'])),
        native_urdf_limited_joint_count=len(limited_indices),native_urdf_position_pass=bool(np.all(q[:,limited_indices]>=native_lower) and np.all(q[:,limited_indices]<=native_upper)),
        tracking_max_rad=maximum(native['q17']-native['qref']),manifold_max_rad=maximum(q-native['q17']@B.T),
        momentum_peak=mp.tolist(),momentum_threshold=(1e-5+.002*me).tolist(),geometry=coverage,transitions=transition)
    if i==0:
        k=99;j=4;lowj=29
        case['failure_mechanism']=dict(interval=native['time'][k:k+2].tolist(),planner_joint=j,native_joint=lowj,
            native_joint_name=mapping[lowj][0],planner_terms={name:float((arr@Bp.T)[k,j]) for name,arr in [('feedforward',ff),('position_feedback',pp),('velocity_feedback',dd),('requested',requested),('actual',lowa),('actual_minus_requested',lowa-requested)]},
            native_terms={name:float(arr[k,lowj]) for name,arr in [('feedforward',ff),('position_feedback',pp),('velocity_feedback',dd),('requested',requested),('actual',lowa),('actual_minus_requested',lowa-requested)]})
    evidence['cases'].append(case);native_total+=len(native['ctrl']);replay_total+=len(replay['ctrl'])
evidence['budget']=dict(native_actual=native_total,replay_actual=replay_total,total_actual=native_total+replay_total,
    native_reserved=budget['native_steps_reserved'],replay_reserved=budget['replay_steps_reserved'],maximum_steps=108000,
    reported_elapsed_s=budget['elapsed_s'],reported_output_bytes=budget['output_bytes'],inflight=budget['inflight'])
residual=max(max(c['max_actual_reference_acceleration_residual_ratio'],c['max_native_reference_acceleration_residual_ratio']) for c in evidence['cases'])
evidence['one_possible_frozen_repair']=dict(mechanism='acceleration reference envelope reserves measured closed-loop tracking headroom',
    observed_max_normalized_residual=residual,engineering_safety_multiplier=2,internal_acceleration_fraction=1-2*residual,
    empirical_hypothesis_not_guarantee=True,original_final_limits_and_27_seconds_unchanged=True)
evidence['paper_claim_audit']='unavailable' if not (ROOT/'paper/PAPER_CLAIM_AUDIT.json').exists() else read(ROOT/'paper/PAPER_CLAIM_AUDIT.json')
timestamp=datetime.now(timezone.utc).isoformat()
(TRACE/'saved_array_audit.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
seen.add(Path(__file__).resolve())
inputs=dict(reviewed_at_utc=timestamp,method='Read-only files and saved-array algebra; no optimizer, simulation, model creation, resume, training, or DEV/TEST.',
    files=[dict(path=str(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p).replace('\\','/'),sha256=hash_file(p),bytes=p.stat().st_size,mtime_utc=datetime.fromtimestamp(p.stat().st_mtime,timezone.utc).isoformat()) for p in sorted(seen)])
(TRACE/'inputs.json').write_text(json.dumps(inputs,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(source=evidence['source_identity'],inputs=evidence['input_identity'],budget=evidence['budget'],cases=[dict(case=c['case'],fit=c['fit_max_rad'],actual_acceleration=c['actual_acceleration']['maximum'],geometry=c['geometry'],replay=c['replay_errors'],initial_id=c['initial_state_id_correct'],desired_errors=[c['desired_path_position_error'],c['desired_rigid_position_error'],c['desired_rigid_rotation_matrix_error']]) for c in evidence['cases']]),ensure_ascii=False,indent=2))

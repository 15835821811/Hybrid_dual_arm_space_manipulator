"""Independent native full-state prediction and read-only candidate gates.

No execute/track/replay calls, trajectory modification, or actual state reset.
The native simulation below is a validator prediction, never execution evidence.
"""
import numpy as np
import mujoco
from scipy.interpolate import BSpline
from scipy.spatial.transform import Rotation
from .contracts import Snapshot
from .trajectory import knots
from .robot import Robot
from .propagation import native_subtree_momentum
from .orchestration import check_budget


def native_command(r,d,q,v,a):
    mujoco.mj_forward(r.model,d);M=r.mass(d);h=d.qfrc_bias-d.qfrc_passive
    b,j=r.base_dofs,r.dof_ids;aa=r.B@a
    ab=-np.linalg.solve(M[np.ix_(b,b)],M[np.ix_(b,j)]@aa+h[b])
    ff=M[np.ix_(j,b)]@ab+M[np.ix_(j,j)]@aa+h[j]
    raw=ff+r.spec.position_kp*(r.B@q-d.qpos[r.qpos_ids])+r.spec.velocity_kd*(r.B@v-d.qvel[j])
    return np.clip(raw,-r.spec.torque_limits,r.spec.torque_limits),raw,ff


def native_prediction(robot,initial,codec,z,dt=.0005,geometry=True):
    r=Robot(physics_dt=dt,scene=robot.scene)
    original_dt=r.model.opt.timestep;r.model.opt.timestep=robot.model.opt.timestep
    if r.compiled_hash()!=robot.model_hash:raise ValueError('FULL_STATE_NATIVE_MODEL_CHANGED')
    r.model.opt.timestep=original_dt
    snap=Snapshot(r.model_hash,initial.mask,initial.state);d=r.restore(snap)
    if np.any(d.qfrc_applied) or np.any(d.xfrc_applied):raise ValueError('UNSUPPORTED_EXTERNAL_WRENCH')
    count=int(round(codec.duration/dt));period=int(round(.002/dt))
    if abs(count*dt-codec.duration)>1e-10 or period<1 or abs(period*dt-.002)>1e-10:raise ValueError('NATIVE_GRID_MISMATCH')
    from .geometry import Geometry
    g=Geometry(r) if geometry else None
    times=np.arange(count+1)*dt;q=codec.evaluate(z,times);v=codec.evaluate(z,times,1);a=codec.evaluate(z,times,2)
    xs=[];vs=[];acc=[];raws=[];nominal=[];gaps=[];momentum=[];volumes=[]
    H=native_subtree_momentum(r,d)
    applied=raw=ff=None;base_residual=0.
    for i in range(count+1):
        if i%64==0:check_budget()
        if i<count and i%period==0:applied,raw,ff=native_command(r,d,q[i],v[i],a[i])
        d.ctrl[r.actuator_ids]=applied;mujoco.mj_forward(r.model,d)
        xs.append(d.qpos.copy());vs.append(d.qvel.copy());acc.append(d.qacc.copy());raws.append(raw.copy());nominal.append(ff.copy())
        if geometry:base_residual=max(base_residual,float(np.max(abs((r.mass(d)@d.qacc+d.qfrc_bias-d.qfrc_passive)[r.base_dofs]))))
        if geometry:
            gaps.append(g.native_scan(d)['minimum']);momentum.append(native_subtree_momentum(r,d,forward=False)-H)
            volumes.append(np.c_[d.geom_xpos,d.geom_xmat*r.model.geom_rbound[:,None]])
        if i<count:mujoco.mj_step(r.model,d)
    return dict(time=times,qpos=np.asarray(xs),velocity=np.asarray(vs),acceleration=np.asarray(acc),raw=np.asarray(raws),nominal=np.asarray(nominal),
                minimum_clearance=min(gaps) if gaps else None,momentum=np.asarray(momentum),volumes=np.asarray(volumes),
                base_residual=base_residual,native_model_hash=r.model_hash,initial_qpos_unchanged=np.array_equal(xs[0],robot.restore(initial).qpos),
                initial_velocity_unchanged=np.array_equal(vs[0],robot.restore(initial).qvel))


def validate_full_state(candidate,snapshot,task,prop,codec,geometry,profile):
    from .validation import input_identity
    before=input_identity(candidate,snapshot,task);r=prop.robot;r.verify_identity()
    if candidate.duration!=task.duration or codec.duration!=task.duration:raise ValueError('DURATION_MISMATCH')
    z=codec.encode(candidate.control);C=candidate.control
    gates=dict(initial_boundary=np.max(abs(codec.evaluate(z,0)-prop.initial_q))<=1e-10 and np.max(abs(codec.evaluate(z,0,1)-prop.initial_v))<=1e-10,
               boundary=np.max(abs(codec.A@C.ravel()-codec.b))<=1e-10,
               q17=bool(np.all(C>=r.lower) and np.all(C<=r.upper)))
    limited=r.model.jnt_limited[r.joints].astype(bool);limits=r.model.jnt_range[r.joints][limited];c67=C@r.B.T
    gates['q67']=bool(np.all(c67[:,limited]>=limits[:,0]) and np.all(c67[:,limited]<=limits[:,1]))
    spline=BSpline(knots(codec.n),C,3)
    gates['velocity']=np.max(abs(spline.derivative(1).c)/candidate.duration/r.spec.planner_velocity_limits)<=1
    gates['acceleration']=np.max(abs(spline.derivative(2).c)/candidate.duration**2/r.spec.planner_acceleration_limits)<=1
    full=native_prediction(r,snapshot,codec,z,.0005);fine=native_prediction(r,snapshot,codec,z,.00025)
    delta=full['volumes']-fine['volumes'][::2];movement=np.diff(full['volumes'],axis=0)
    err=float(np.max(np.linalg.norm(delta[:,:,:3],axis=-1)+np.linalg.norm(delta[:,:,3:],axis=-1)))
    spatial=float(np.max(np.linalg.norm(movement[:,:,:3],axis=-1)+np.linalg.norm(movement[:,:,3:],axis=-1)))
    gates['propagation_and_refinement']=err<=profile['prop_finite_volume_m'] and spatial<=.002
    gates['exact_initial_state']=full['initial_qpos_unchanged'] and full['initial_velocity_unchanged']
    q67=full['qpos'][:,r.qpos_ids];q17=q67@r.Bplus.T
    gates['actual_q17']=bool(np.all(q17>=r.lower) and np.all(q17<=r.upper))
    gates['actual_q67']=bool(np.all(q67[:,limited]>=limits[:,0]) and np.all(q67[:,limited]<=limits[:,1]))
    gates['actual_velocity']=np.max(abs(full['velocity'][:,r.dof_ids])/(abs(r.B)@r.spec.planner_velocity_limits))<=1
    gates['actual_acceleration']=np.max(abs(full['acceleration'][:,r.dof_ids])/(abs(r.B)@r.spec.planner_acceleration_limits))<=1
    gates['unactuated_equation']=full['base_residual']<=1e-8
    gates['planner_actual_velocity']=np.max(abs(full['velocity'][:,r.dof_ids]@r.Bplus.T)/r.spec.planner_velocity_limits)<=1
    gates['planner_actual_acceleration']=np.max(abs(full['acceleration'][:,r.dof_ids]@r.Bplus.T)/r.spec.planner_acceleration_limits)<=1
    gates['geometry']=full['minimum_clearance']>=profile['physical_geometry_clearance_m']
    tau=float(np.max(abs(full['nominal'])/r.spec.torque_limits));raw=float(np.max(abs(full['raw'])/r.spec.torque_limits))
    gates['actuator']=tau<=.8;gates['raw_torque']=raw<=1.
    P=float(np.max(np.linalg.norm(full['momentum'][:,:3],axis=1)));L=float(np.max(np.linalg.norm(full['momentum'][:,3:],axis=1)))
    gates['world_momentum']=P<=profile['momentum_linear_kg_m_s'] and L<=profile['momentum_angular_kg_m2_s']
    errors=[];d=r.new_data()
    for target in task.poses:
        index=int(round(target.time/.0005))
        if abs(index*.0005-target.time)>1e-10:raise ValueError('TASK_TIME_NOT_ON_NATIVE_GRID')
        d.qpos[:]=full['qpos'][index];mujoco.mj_kinematics(r.model,d)
        body=r.tips[target.tip];R=d.xmat[body].reshape(3,3);pos=d.xpos[body].copy()
        if target.tip==0:pos+=R@np.array([.0475,0,0])
        pe=float(np.linalg.norm(pos-target.position));re=float(Rotation.from_matrix(target.rotation.T@R).magnitude())
        errors.append(dict(time=target.time,tip=target.tip,position_m=pe,rotation_rad=re,passed=(not target.position_enabled or pe<=target.position_tolerance) and (not target.rotation_enabled or re<=target.rotation_tolerance)))
    gates['task']=all(x['passed'] for x in errors)
    gates={k:bool(v) for k,v in gates.items()};failures=[k for k,v in gates.items() if not v]
    if before!=input_identity(candidate,snapshot,task):raise RuntimeError('VALIDATOR_MUTATED_INPUT')
    r.verify_identity()
    return dict(status='PLAN_VALID' if not failures else 'REJECTED',gates=gates,failures=failures,input_hash=before,input_unchanged=True,
                output_candidate_hash=candidate.identity,task_errors=errors,max_nominal_torque_ratio=tau,max_raw_torque_ratio=raw,
                minimum_clearance=full['minimum_clearance'],momentum_linear=P,momentum_angular=L,propagation_volume_error_m=err,
                geometry_step_m=spatial,continuous_time_collision_proof=False,scope='INDEPENDENT_NATIVE_PREDICTION_NOT_EXECUTION',
                native_steps=len(full['time'])+len(fine['time'])-2)

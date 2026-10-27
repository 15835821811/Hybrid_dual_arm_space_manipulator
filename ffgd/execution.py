"""Native 67-torque reference tracking and independent input replay.

Controller equation migrated explicitly from pinned floating_ff_torque_pd_v1.
Safety retains torque limits, geometry/limit checks, nonfinite/warning aborts.
Only mj_step updates physical state after initial restoration.
"""
import json
import numpy as np
import mujoco
import os
from .trajectory import basis
from .geometry import Geometry
from .orchestration import check_budget,atomic_json


def command(robot,data,q,qd,qdd):
    r=robot;m=r.model;mujoco.mj_forward(m,data);M=r.mass(data)
    a,b=r.dof_ids,r.base_dofs;h=data.qfrc_bias-data.qfrc_passive
    qa,va,aa=r.B@q,r.B@qd,r.B@qdd
    ab=-np.linalg.solve(M[np.ix_(b,b)],M[np.ix_(b,a)]@aa+h[b])
    ff=M[np.ix_(a,b)]@ab+M[np.ix_(a,a)]@aa+h[a]
    raw=ff+r.spec.position_kp*(qa-data.qpos[r.qpos_ids])+r.spec.velocity_kd*(va-data.qvel[a])
    return np.clip(raw,-r.spec.torque_limits,r.spec.torque_limits),raw


def refine_physics(robot,snapshot,substeps):
    """Rebind identical complete state to a model differing only in timestep."""
    from .robot import Robot
    from .contracts import Snapshot
    if type(substeps) is not int or substeps not in (1,2,4):raise ValueError('UNSUPPORTED_PHYSICS_SUBSTEPS')
    robot.verify_identity()
    if snapshot.model_hash!=robot.model_hash:raise ValueError('snapshot model identity mismatch')
    if substeps==1:return robot,snapshot
    dt=float(robot.model.opt.timestep);refined=Robot(physics_dt=dt/substeps,scene=robot.scene)
    refined.model.opt.timestep=dt
    if refined.compiled_hash()!=robot.model_hash:raise ValueError('MODEL_DIFFERENCE_BEYOND_TIMESTEP')
    refined.model.opt.timestep=dt/substeps
    rebound=Snapshot(refined.model_hash,snapshot.mask,snapshot.state.copy())
    np.testing.assert_array_equal(rebound.state,snapshot.state)
    refined.restore(rebound)  # Validate full mask/shape without resetting fields.
    return refined,rebound


def track(robot,snapshot,candidate,output,profile,control_substeps=1):
    from .propagation import native_momentum
    r=robot;m=r.model;data=r.restore(snapshot);geometry=Geometry(r);dt=m.opt.timestep
    if type(control_substeps) is not int or control_substeps not in (1,2,4):raise ValueError('UNSUPPORTED_PHYSICS_SUBSTEPS')
    steps=int(round(candidate.duration/dt))
    if abs(steps*dt-candidate.duration)>1e-10:raise ValueError('duration is not integral native steps')
    if steps%control_substeps:raise ValueError('duration is not integral control periods')
    states=[];raws=[];controls=[];forces=[];external=[];momentum=[];clearance=[];residual=[];tracking=[]
    status='FINISHED';initial_H=native_momentum(r,data);selected_hash=candidate.identity
    def recorded_bundle():
        return dict(states=np.asarray(states),ctrl=np.asarray(controls),raw_torque=np.asarray(raws),qfrc_applied=np.asarray(forces),
                    xfrc_applied=np.asarray(external),momentum_residual=np.asarray(momentum),clearance=np.asarray(clearance),
                    manifold_residual=np.asarray(residual),tracking_q67=np.asarray(tracking),
                    initial_state=snapshot.state,mask=snapshot.mask,control_points=candidate.control,duration=candidate.duration,
                    physics_dt=dt,control_substeps=control_substeps)
    def checkpoint(filename):
        temporary=output/(filename+'.tmp')
        with temporary.open('wb') as stream:np.savez_compressed(stream,**recorded_bundle())
        os.replace(temporary,output/filename)
    for i in range(steps+1):
        mujoco.mj_forward(m,data)
        states.append(r.snapshot(data).state.copy())
        scan=geometry.native_scan(data);clearance.append(scan['minimum'])
        momentum.append(native_momentum(r,data)-initial_H)
        low=data.qpos[r.qpos_ids];q17=r.Bplus@low
        residual.append(low-r.B@q17)
        q=basis(len(candidate.control),i*dt,candidate.duration)@candidate.control
        tracking.append(low-r.B@q)
        if i%500==0:
            checkpoint('execution_partial.npz')
            atomic_json(output/'execution_progress.json',dict(status='RUNNING',native_steps=len(controls),planned_steps=steps,candidate_hash=selected_hash))
        if status!='FINISHED':break
        if i%64==0:
            try:check_budget()
            except RuntimeError as error:
                if 'RESOURCE_' not in str(error):raise
                status='RESOURCE_BUDGET_ABORT';break
        if scan['minimum']<profile['physical_geometry_clearance_m']:
            status='SAFETY_ABORT_GEOMETRY';break
        limited=m.jnt_limited[r.joints].astype(bool);bounds=m.jnt_range[r.joints]
        if np.any(low[limited]<bounds[limited,0]) or np.any(low[limited]>bounds[limited,1]):
            status='SAFETY_ABORT_JOINT';break
        if np.any(q17<r.lower) or np.any(q17>r.upper):
            status='SAFETY_ABORT_PLANNER_JOINT';break
        if i==steps:break
        if i%control_substeps==0:
            qd=basis(len(candidate.control),i*dt,candidate.duration,1)@candidate.control
            qdd=basis(len(candidate.control),i*dt,candidate.duration,2)@candidate.control
            applied,raw=command(r,data,q,qd,qdd)
        else:
            # Hold the previous control; preserve the exact replay observation schedule.
            mujoco.mj_forward(m,data)
        if not np.isfinite(raw).all():status='NONFINITE_COMMAND';break
        data.ctrl[r.actuator_ids]=applied
        raws.append(raw);controls.append(data.ctrl.copy());forces.append(data.qfrc_applied.copy());external.append(data.xfrc_applied.copy())
        mujoco.mj_step(m,data)
        # Record the post-input native state on the next loop before stopping,
        # so an interrupted/failed execution still has N+1 states for N inputs.
        if np.any(data.warning.number):status='SIMULATOR_WARNING'
    bundle=recorded_bundle();checkpoint('execution.npz')
    atomic_json(output/'execution_progress.json',dict(status=status,native_steps=len(controls),planned_steps=steps,candidate_hash=selected_hash))
    summary=dict(status=status,planned_steps=steps,actual_steps=len(controls),model_hash=r.model_hash,candidate_hash=selected_hash,
                 controller='floating_ff_torque_pd_v1',physics_dt=dt,control_period=dt*control_substeps,control_substeps=control_substeps,
                 controller_updates=len(controls[::control_substeps]),state_writes_after_restore=0,geometry_safety='abort on exact native penetration at every physics substep',
                 min_clearance=float(min(clearance)),raw_torque_ratio=float(np.max(abs(bundle['raw_torque'])/r.spec.torque_limits)) if len(raws) else None,
                 clipped_control_steps=int(np.count_nonzero(np.any(abs(bundle['raw_torque'])>r.spec.torque_limits,axis=1))) if len(raws) else 0,
                 safety_abort=status.startswith('SAFETY_ABORT'),task_based_cartesian_assist=False,
                 linear_momentum_residual=float(np.max(np.linalg.norm(bundle['momentum_residual'][:,:3],axis=1))),
                 angular_momentum_residual=float(np.max(np.linalg.norm(bundle['momentum_residual'][:,3:],axis=1))),
                 max_manifold_residual=float(np.max(abs(bundle['manifold_residual']))),max_joint_tracking=float(np.max(abs(bundle['tracking_q67']))))
    (output/'execution.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    assert candidate.identity==selected_hash
    return bundle,summary


def replay(robot,bundle,snapshot):
    """Restore only once, replay input sequence; never inject intermediate state."""
    r=robot;m=r.model;d=r.restore(snapshot);errors=[]
    mujoco.mj_forward(m,d)
    for i in range(len(bundle['ctrl'])+1):
        # Reproduce the controller's observation schedule; it updates derived/warmstart quantities.
        mujoco.mj_forward(m,d)
        errors.append(float(np.max(abs(r.snapshot(d).state-bundle['states'][i]))))
        if i==len(bundle['ctrl']):break
        # native_momentum in recorder and command also call forward; match observations.
        mujoco.mj_forward(m,d);mujoco.mj_forward(m,d)
        d.ctrl[:]=bundle['ctrl'][i];d.qfrc_applied[:]=bundle['qfrc_applied'][i];d.xfrc_applied[:]=bundle['xfrc_applied'][i]
        mujoco.mj_step(m,d)
    return dict(max_state_abs=max(errors),steps=len(bundle['ctrl']),state_restore_calls=1,per_frame_state_injection=False)

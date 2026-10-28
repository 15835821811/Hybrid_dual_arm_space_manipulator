"""Public prescribed target force, shared units with independent native inputs.

Only a separate free target is supported. No robot generalized forces, state
edits, trajectory lookup, feedback policy or scenario identifiers are accepted.
"""
import numpy as np
import mujoco


def target_indices(robot):
    joint=robot.model.joint(robot.spec.target_free_joint_name)
    body=int(robot.model.jnt_bodyid[joint.id]);adr=int(joint.dofadr[0])
    if robot.model.jnt_type[joint.id]!=mujoco.mjtJoint.mjJNT_FREE or robot.model.body_parentid[body]!=0:raise ValueError('FORCING_REQUIRES_SEPARATE_FREE_TARGET')
    if np.any(robot.model.body_ipos[body]) or np.any(robot.model.dof_damping[adr:adr+6]):raise ValueError('TARGET_FORCE_REQUIRES_COM_ORIGIN_AND_ZERO_FREE_DAMPING')
    return body,adr


def target_force(pulses,time):
    """N, with task-relative seconds. Endpoints and all outside times are zero."""
    t=np.asarray(time,dtype=float);force=np.zeros(t.shape+(3,))
    for pulse in pulses:
        phase=np.clip((t-pulse.start)/(pulse.end-pulse.start),0.,1.)
        envelope=np.where((t>pulse.start)&(t<pulse.end),.5*(1-np.cos(2*np.pi*phase)),0.)
        force+=envelope[...,None]*pulse.peak_force
    return force


def generalized_forces(robot,pulses,times):
    """A force at COM has translation generalized coordinates for a free body."""
    result=np.zeros(np.asarray(times).shape+(robot.model.nv,))
    if pulses:
        _,adr=target_indices(robot);result[...,adr:adr+3]=target_force(pulses,times)
    return result


def apply_native_target_force(robot,data,pulses,time):
    """Set declared input only; never write qpos, qvel or integration state."""
    body,_=target_indices(robot)
    data.xfrc_applied[body,:3]=target_force(pulses,time)


def target_displacement(robot,pulses,time):
    """Closed-form translation due solely to the declared force since t=0."""
    body,_=target_indices(robot);mass=float(robot.model.body_mass[body]);value=np.zeros(3)
    for pulse in pulses:
        length=pulse.end-pulse.start;s=float(np.clip(time-pulse.start,0.,length));w=2*np.pi/length
        twice_integrated=.25*s*s+.5*(np.cos(w*s)-1)/(w*w)
        coast=max(0.,time-pulse.end)*length*.5
        value+=pulse.peak_force*(twice_integrated+coast)/mass
    return value


def inspect_target_inputs(robot,bundle,pulses):
    """Read saved actual states/inputs; check independent impulse and work laws."""
    from .contracts import Snapshot
    r=robot;m=r.model;body,adr=target_indices(r);dt=float(m.opt.timestep)
    states=bundle['states'];n=len(bundle['ctrl']);mass=float(m.body_mass[body])
    if len(states)!=n+1:raise ValueError('INCOMPLETE_FORCE_LEDGER')
    positions=[];velocities=[];angular=[];kinetic=[]
    for state in states:
        d=r.restore(Snapshot(r.model_hash,int(bundle['mask']),state));mujoco.mj_forward(m,d)
        v=d.qvel[adr:adr+3];omega=d.qvel[adr+3:adr+6];R=d.xmat[body].reshape(3,3)
        pos=d.xipos[body].copy();P=mass*v;L=R@(m.body_inertia[body]*omega)+np.cross(pos,P)
        positions.append(pos);velocities.append(v.copy());angular.append(L);kinetic.append(.5*mass*(v@v)+.5*((m.body_inertia[body]*omega)@omega))
    positions=np.asarray(positions);velocities=np.asarray(velocities);angular=np.asarray(angular);kinetic=np.asarray(kinetic)
    actual=np.asarray(bundle['xfrc_applied']);expected=np.zeros_like(actual)
    expected[:,body,:3]=target_force(pulses,(np.arange(n)+.5)*dt)
    F=actual[:,body,:3]
    impulse=np.vstack((np.zeros(3),np.cumsum(F*dt,axis=0)))
    angular_impulse=np.vstack((np.zeros(3),np.cumsum(np.cross(positions[:-1],F)*dt,axis=0)))
    work=np.r_[0.,np.cumsum(np.sum(F*(velocities[:-1]+velocities[1:])*.5,axis=1)*dt)]
    linear=float(np.max(abs(mass*(velocities-velocities[0])-impulse)))
    spin=float(np.max(abs(angular-angular[0]-angular_impulse)))
    work_error=float(np.max(abs(kinetic-kinetic[0]-work)))
    duration=n*dt
    analytic=positions[0]+duration*velocities[0]+target_displacement(r,pulses,duration)
    position_error=float(np.linalg.norm(positions[-1]-analytic))
    input_error=float(np.max(abs(actual-expected))) if n else None
    coast_start=max(p.end for p in pulses)
    coast=velocities[np.arange(n+1)*dt>=coast_start]
    coast_error=float(np.max(abs(coast-coast[0]))) if len(coast) else None
    gates=dict(declared_force_inputs=input_error is not None and input_error<=1e-14 and not np.any(bundle['qfrc_applied']),
        target_linear_impulse=linear<=1e-9,target_angular_impulse=spin<=1e-9,target_work_balance=work_error<=1e-9,
        target_analytic_position=position_error<=3e-6,target_coast_velocity=coast_error is not None and coast_error<=1e-10)
    return dict(passed=all(gates.values()),gates=gates,steps=n,linear_balance_max_abs=linear,angular_balance_max_abs=spin,
        work_balance_max_abs=work_error,maximum_input_error=input_error,analytic_final_position_error=position_error,
        coast_velocity_max_error=coast_error,impulse=impulse[-1].tolist(),final_target_velocity=velocities[-1].tolist(),
        initial_target_velocity=velocities[0].tolist(),source='SAVED_ACTUAL_STATES_AND_INPUTS',velocity_injection=False,
        force_sampling='native interval midpoint; zeros outside declared pulse',scope='Separate target COM force protocol; no robot momentum reset')

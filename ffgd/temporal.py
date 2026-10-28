"""Public interval semantics, independent of scenario, waypoint number or seed.

Target-frame requests are transformed by predicted/actual target body state.
Condition tokens use a declared free-body prediction, never future observations.
"""
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from .contracts import PoseConstraint


def target_body(robot):
    joint=robot.model.joint(robot.spec.target_free_joint_name)
    return int(robot.model.jnt_bodyid[joint.id])


def world_goal(interval,positions,rotations,body):
    if interval.frame=='world':return interval.position,interval.rotation
    return positions[body]+rotations[body]@interval.position,rotations[body]@interval.rotation


def interpolate_qpos(robot,times,qpos,t):
    """Read-only physical-state interpolation for non-grid task timestamps."""
    if t<times[0] or t>times[-1]:raise ValueError('TASK_TIME_OUTSIDE_RECORDED_STATES')
    right=min(int(np.searchsorted(times,t)),len(times)-1);left=max(0,right-1)
    if times[right]==t:return qpos[right].copy(),0.
    dt=times[right]-times[left];alpha=0. if dt==0 else (t-times[left])/dt
    x=(1-alpha)*qpos[left]+alpha*qpos[right]
    # Slerp every free/ball quaternion; linear position/hinge interpolation.
    for joint in range(robot.model.njnt):
        typ=robot.model.jnt_type[joint];qa=int(robot.model.jnt_qposadr[joint])
        if typ not in (mujoco.mjtJoint.mjJNT_FREE,mujoco.mjtJoint.mjJNT_BALL):continue
        adr=qa+3 if typ==mujoco.mjtJoint.mjJNT_FREE else qa
        a=qpos[left,adr:adr+4];b=qpos[right,adr:adr+4]
        if np.dot(a,b)<0:b=-b
        cosine=np.clip(a@b,-1.,1.);angle=np.arccos(cosine)
        value=(1-alpha)*a+alpha*b if angle<1e-7 else (np.sin((1-alpha)*angle)*a+np.sin(alpha*angle)*b)/np.sin(angle)
        x[adr:adr+4]=value/np.linalg.norm(value)
    return x,float(dt)


def condition_poses(robot,snapshot,task):
    """Three sparse interval samples in the trained timed-pose token schema.

    This predictor is valid for the unforced spherical-inertia target currently
    present in the model. Refuse unsupported angular dynamics explicitly.
    """
    if not task.intervals:return task.poses
    d=robot.restore(snapshot);m=robot.model;body=target_body(robot)
    joint=m.joint(robot.spec.target_free_joint_name);qa=int(joint.qposadr[0]);va=int(joint.dofadr[0])
    if np.any(d.qfrc_applied) or np.any(d.xfrc_applied):raise ValueError('INTERVAL_CONDITION_EXTERNAL_FORCE_UNSUPPORTED')
    if not np.allclose(m.body_inertia[body],m.body_inertia[body,0],rtol=1e-8,atol=1e-12):raise ValueError('INTERVAL_CONDITION_ANISOTROPIC_TARGET_UNSUPPORTED')
    R0=Rotation.from_quat(d.qpos[qa+3:qa+7][[1,2,3,0]])
    result=list(task.poses)
    for interval in task.intervals:
        for time in (interval.start,(interval.start+interval.end)/2,interval.end):
            pos,rot=interval.position,interval.rotation
            if interval.frame=='target':
                R=(R0*Rotation.from_rotvec(d.qvel[va+3:va+6]*time)).as_matrix()
                pos=d.qpos[qa:qa+3]+d.qvel[va:va+3]*time+R@pos;rot=R@rot
            result.append(PoseConstraint(time,interval.tip,pos,rot,interval.position_tolerance,interval.rotation_tolerance,interval.position_enabled,interval.rotation_enabled))
    return tuple(sorted(result,key=lambda p:(p.time,p.tip,tuple(p.position),tuple(p.rotation.ravel()))))


def inspect_intervals(robot,task,times,qpos):
    """Independent native FK at every supplied state inside each interval.

    Off-grid endpoints use an explicit interpolation, with its maximum temporal
    bracket reported. No interval is reduced to arrival-only acceptance.
    """
    if not task.intervals:return []
    times=np.asarray(times);qpos=np.asarray(qpos);r=robot;m=r.model;body=target_body(r);d=mujoco.MjData(m)
    reports=[]
    for interval in task.intervals:
        grid=np.unique(np.r_[interval.start,times[(times>=interval.start)&(times<=interval.end)],interval.end])
        complete=bool(times[0]<=interval.start and times[-1]>=interval.end)
        maximum_p=maximum_r=bracket=0.;worst_p=worst_r=None
        for t in grid:
            if not times[0]<=t<=times[-1]:continue
            d.qpos[:],dt=interpolate_qpos(r,times,qpos,t)
            mujoco.mj_kinematics(m,d)
            rotations=d.xmat.reshape(-1,3,3);P,R=world_goal(interval,d.xpos,rotations,body)
            tip=r.tips[interval.tip];pos=d.xpos[tip].copy()
            if interval.tip==0:pos+=rotations[tip]@np.array([.0475,0,0])
            pe=float(np.linalg.norm(pos-P));re=float(Rotation.from_matrix(R.T@rotations[tip]).magnitude())
            if pe>=maximum_p:maximum_p=pe;worst_p=float(t)
            if re>=maximum_r:maximum_r=re;worst_r=float(t)
            bracket=max(bracket,float(dt))
        reports.append(dict(start=interval.start,end=interval.end,tip=interval.tip,frame=interval.frame,samples=len(grid),
            maximum_position_m=maximum_p,maximum_rotation_rad=maximum_r,worst_position_time=worst_p,worst_rotation_time=worst_r,
            endpoint_interpolation_bracket_s=bracket,complete=complete,
            passed=complete and (not interval.position_enabled or maximum_p<=interval.position_tolerance) and (not interval.rotation_enabled or maximum_r<=interval.rotation_tolerance),
            scope='NATIVE_GRID_INTERVAL_WITH_INTERPOLATED_ENDPOINTS_NOT_CONTINUOUS_PROOF'))
    return reports

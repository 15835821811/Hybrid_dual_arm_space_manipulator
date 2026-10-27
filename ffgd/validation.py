"""Independent read-only native candidate validation. No repair or solver calls."""
import hashlib
import json
import numpy as np
import mujoco
from scipy.interpolate import BSpline
from scipy.spatial.transform import Rotation
from .trajectory import knots
from .contracts import array_hash


def input_identity(candidate,snapshot,task):
    records=[dict(time=p.time,tip=p.tip,position=p.position.tolist(),rotation=p.rotation.tolist(),
                  position_tolerance=p.position_tolerance,rotation_tolerance=p.rotation_tolerance,
                  position_enabled=p.position_enabled,rotation_enabled=p.rotation_enabled) for p in task.poses]
    return hashlib.sha256((candidate.identity+snapshot.model_hash+array_hash(snapshot.state)+json.dumps(dict(duration=task.duration,poses=records),sort_keys=True)).encode()).hexdigest()


def validate(candidate,snapshot,task,propagator,codec,geometry,profile,max_intervals=2048):
    before=input_identity(candidate,snapshot,task);r=propagator.robot;m=r.model
    r.verify_identity()
    failures=[];gates={};report=dict(status='INCOMPLETE_VALIDATION',input_hash=before,gates=gates)
    if candidate.duration!=task.duration or codec.duration!=task.duration:raise ValueError('duration identity mismatch')
    z=codec.encode(candidate.control)
    if snapshot.model_hash!=r.model_hash:raise ValueError('snapshot robot identity mismatch')
    if not np.array_equal(r.restore(snapshot).qpos,propagator.qpos.numpy()):raise ValueError('propagator snapshot mismatch')
    gates['initial_boundary']=bool(np.max(abs(codec.evaluate(z,0)-propagator.initial_q))<=1e-10 and np.max(abs(codec.evaluate(z,0,1)-propagator.initial_v))<=1e-10)
    gates['boundary']=bool(np.max(abs(codec.A@candidate.control.ravel()-codec.b))<=1e-10)
    # B-spline convex hull bounds include all time, not just the sampled knots.
    C=candidate.control
    gates['q17']=bool(np.all(C>=r.lower) and np.all(C<=r.upper))
    C67=C@r.B.T;limited=m.jnt_limited[r.joints].astype(bool);limits=m.jnt_range[r.joints][limited]
    gates['q67']=bool(np.all(C67[:,limited]>=limits[:,0]) and np.all(C67[:,limited]<=limits[:,1]))
    spline=BSpline(knots(codec.n),C,3)
    gates['velocity']=bool(np.max(abs(spline.derivative(1).c)/candidate.duration/r.spec.planner_velocity_limits)<=1)
    gates['acceleration']=bool(np.max(abs(spline.derivative(2).c)/candidate.duration**2/r.spec.planner_acceleration_limits)<=1)
    previous=None;converged=False;last=None;propagation_mass_calls=0
    intervals=32
    while intervals<=max_intervals:
        grid=np.unique(np.r_[np.linspace(0,candidate.duration,intervals+1),knots(codec.n)*candidate.duration,[p.time for p in task.poses]])
        xs,vs=propagator.native_rollout(codec,z,intervals,grid=grid)
        propagation_mass_calls+=2*len(grid)-1
        d=mujoco.MjData(m);positions=[];rotations=[]
        for x in xs:
            d.qpos[:]=x;mujoco.mj_kinematics(m,d);positions.append(d.geom_xpos.copy());rotations.append(d.geom_xmat.reshape(-1,3,3).copy())
        P,R=np.asarray(positions),np.asarray(rotations)
        spatial_step=float(np.max(np.linalg.norm(np.diff(P,axis=0),axis=-1)+np.linalg.norm(np.diff(R,axis=0),axis=(-1,-2))*m.geom_rbound))
        error=None
        if previous is not None:
            oldgrid,oldP,oldR=previous;indices=np.searchsorted(grid,oldgrid)
            if not np.allclose(grid[indices],oldgrid,atol=1e-12):raise RuntimeError('refinement lost old grid')
            error=float(np.max(np.linalg.norm(P[indices]-oldP,axis=-1)+np.linalg.norm(R[indices]-oldR,axis=(-1,-2))*m.geom_rbound))
        last=(grid,xs,vs)
        if error is not None and error<=profile['prop_finite_volume_m'] and spatial_step<=.002:
            converged=True;break
        previous=(grid,P,R);intervals*=2
    gates['propagation_and_refinement']=converged
    report.update(intervals=min(intervals,max_intervals),next_required_intervals=None if converged else intervals,geometry_step_m=spatial_step,propagation_volume_error_m=error,
                  native_propagation_mass_calls=propagation_mass_calls)
    if not converged:
        assert before==input_identity(candidate,snapshot,task)
        report['input_unchanged']=True
        r.verify_identity()
        return report
    grid,xs,vs=last;d=mujoco.MjData(m);min_clearance=10.;minimum_geometry_location=None;max_torque=0.;base_residual=0.;task_errors=[];momentum_linear=0.;momentum_angular=0.
    from .propagation import native_subtree_momentum
    accelerations=codec.evaluate(z,grid,2)
    distance_queries=0
    for i,t in enumerate(grid):
        d.qpos[:]=xs[i];d.qvel[:]=vs[i];mujoco.mj_forward(m,d)
        momentum=native_subtree_momentum(r,d,forward=False)-propagator.H.numpy()
        momentum_linear=max(momentum_linear,float(np.linalg.norm(momentum[:3])))
        momentum_angular=max(momentum_angular,float(np.linalg.norm(momentum[3:])))
        scan=geometry.native_scan(d)
        if scan['minimum']<min_clearance:
            min_clearance=scan['minimum'];minimum_geometry_location=dict(time=float(t),pair=scan['pair'],
                geom_names=[m.geom(g).name for g in scan['pair']] if scan['pair'] is not None else [])
        distance_queries+=scan['queries']
        M=r.mass(d);h=d.qfrc_bias-d.qfrc_passive;a,b=r.dof_ids,r.base_dofs
        aa=r.B@accelerations[i];ab=-np.linalg.solve(M[np.ix_(b,b)],M[np.ix_(b,a)]@aa+h[b])
        tau=M[np.ix_(a,b)]@ab+M[np.ix_(a,a)]@aa+h[a]
        max_torque=max(max_torque,float(np.max(abs(tau)/r.spec.torque_limits)))
        base_residual=max(base_residual,float(np.max(abs(M[np.ix_(b,b)]@ab+M[np.ix_(b,a)]@aa+h[b]))))
        for target in task.poses:
            if target.time!=t:continue
            body=r.tips[target.tip];RR=d.xmat[body].reshape(3,3);pos=d.xpos[body].copy()
            if target.tip==0:pos+=RR@np.array([.0475,0,0])
            pe=float(np.linalg.norm(pos-target.position));re=float(Rotation.from_matrix(target.rotation.T@RR).magnitude())
            task_errors.append(dict(time=float(t),tip=target.tip,position_m=pe,rotation_rad=re,
                                    passed=(not target.position_enabled or pe<=target.position_tolerance) and (not target.rotation_enabled or re<=target.rotation_tolerance)))
    gates.update(geometry=min_clearance>=profile['physical_geometry_clearance_m'],actuator=max_torque<=.8,
                 unactuated_equation=base_residual<=1e-8,task=all(e['passed'] for e in task_errors) and len(task_errors)==len(task.poses),
                 world_momentum=momentum_linear<=profile['momentum_linear_kg_m_s'] and momentum_angular<=profile['momentum_angular_kg_m2_s'])
    failures=[k for k,v in gates.items() if not v]
    report.update(status='PLAN_VALID' if not failures else 'REJECTED',failures=failures,minimum_clearance=min_clearance,
                  minimum_geometry_location=minimum_geometry_location,
                  max_nominal_torque_ratio=max_torque,base_residual=base_residual,task_errors=task_errors,
                  continuous_time_collision_proof=False,grid_points=len(grid),output_candidate_hash=candidate.identity)
    report.update(momentum_linear=momentum_linear,momentum_angular=momentum_angular)
    report.update(native_geometry_distance_queries=distance_queries,native_actuation_mass_calls=len(grid))
    after=input_identity(candidate,snapshot,task)
    if before!=after:raise RuntimeError('VALIDATOR_MUTATED_INPUT')
    report['input_unchanged']=True
    r.verify_identity()
    return report

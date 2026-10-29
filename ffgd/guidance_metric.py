"""Full-rank Sobolev metric for a denoising gradient, never trajectory smoothing.

I + weight * integral(A''^T A'')/T is SPD on every normalized free coordinate.
It changes a descent direction inside a sampling step; no control point is fixed
and no path optimization or post-sampling correction is performed here.
"""
import numpy as np
import torch
from .orchestration import check_budget
from .trajectory import basis


def acceleration_metric(codec,scale,limits,grid,weight):
    if not np.isfinite(weight) or weight<0:raise ValueError('invalid metric weight')
    times=np.asarray(grid);dt=np.diff(times)
    quadrature=np.r_[dt[0]/2,(dt[:-1]+dt[1:])/2,dt[-1]/2]/codec.duration
    controls=codec.N.reshape(codec.n,codec.dof,codec.free_dim)*np.asarray(scale)[None,None,:]
    operator=np.einsum('tn,njf->tjf',basis(codec.n,times,codec.duration,2),controls)/np.asarray(limits)[None,:,None]
    operator=operator*np.sqrt(quadrature[:,None,None]/codec.dof)
    A=operator.reshape(-1,codec.free_dim)
    return torch.tensor(np.eye(codec.free_dim)+weight*(A.T@A),dtype=torch.float64)


def residual_jacobian(residual,variable,observer=None,batch_size=1):
    """Exact whole-graph reverse derivatives; no terminal-only approximation."""
    if residual is None or residual.ndim!=1:raise ValueError('missing task residual vector')
    if not 1<=batch_size<=12:raise ValueError('invalid Jacobian VJP batch size')
    rows=[]
    for index in range(0,len(residual),batch_size):
        check_budget()
        end=min(index+batch_size,len(residual))
        if batch_size==1:
            block=torch.autograd.grad(residual[index],variable,retain_graph=True)[0][None]
        else:
            cotangent=torch.eye(len(residual),dtype=residual.dtype,device=residual.device)[index:end]
            block=torch.autograd.grad(residual,variable,grad_outputs=cotangent,retain_graph=True,is_grads_batched=True)[0]
        rows.append(block.detach())
        if observer:observer(end,len(residual))
    return torch.cat(rows)


def instantaneous_tip_jacobians(robot, data):
    """Local momentum-conserving velocity map; NOT a trajectory propagator.

    The actual whole-path gradient remains authoritative. This approximation
    is only an SPD direction metric: it omits propagation history and servo
    lag, and may be rejected by the independent energy trial.
    """
    import mujoco
    r = robot; m = r.model
    mujoco.mj_forward(m, data)
    M = r.mass(data)
    G = np.zeros((m.nv, 17))
    G[r.dof_ids] = r.B
    G[r.base_dofs] = -np.linalg.solve(M[np.ix_(r.base_dofs, r.base_dofs)], M[np.ix_(r.base_dofs, r.dof_ids)] @ r.B)
    jacobians = []
    for tip, body in enumerate(r.tips):
        point = data.xpos[body].copy()
        if tip == 0:point += data.xmat[body].reshape(3, 3) @ np.array([.0475, 0, 0])
        jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
        mujoco.mj_jac(m, data, jp, jr, point, body)
        jacobians.append(np.concatenate((jp @ G, jr @ G), axis=0))
    return np.asarray(jacobians), G


def instantaneous_collision_rows(robot, data, geometry, guide, velocity_map, buffer):
    """Spatial support-gap derivatives through the local velocity map only."""
    import mujoco
    P=torch.tensor(data.xpos.copy(),dtype=torch.float64,requires_grad=True)
    R=torch.tensor(data.xmat.reshape(-1,3,3).copy(),dtype=torch.float64,requires_grad=True)
    gaps=guide.gaps(dict(position=P,rotation=R))
    penalties=torch.relu((buffer-gaps)/buffer).square().detach().numpy()
    ids=np.flatnonzero(penalties>0); rows=[]; body_jacobians={}
    for index in ids:
        check_budget()
        gp,gr=torch.autograd.grad(gaps[int(index)],(P,R),retain_graph=True)
        gp=gp.numpy();gr=gr.numpy()
        angular=np.cross(R.detach().numpy().transpose(0,2,1),gr.transpose(0,2,1)).sum(axis=1)
        active=np.flatnonzero(np.any(gp!=0,axis=1)|np.any(angular!=0,axis=1))
        row=np.zeros(17)
        for body in active:
            if int(body) not in body_jacobians:
                jp=np.zeros((3,robot.model.nv));jr=np.zeros_like(jp)
                mujoco.mj_jac(robot.model,data,jp,jr,data.xpos[body],int(body))
                body_jacobians[int(body)]=(jp@velocity_map,jr@velocity_map)
            jp,jr=body_jacobians[int(body)];row+=gp[body]@jp+angular[body]@jr
        rows.append(row)
    return ids,np.asarray(rows).reshape(-1,17),penalties


def instantaneous_task_metric(robot, codec, scale, task, grid, qpos, task_weight=1., task_fraction=1., task_objective='center_squared',
                              geometry=None,collision_weight=1.,buffer=.002,collision_aggregation='pair_mean'):
    """Approximate task curvature at every declared interval grid point.

    Includes both interval quadrature and its worst instant. It acts on all
    free variables through the same boundary codec. It returns a PSD addition
    to the existing SPD metric, never a path or new target.
    """
    import mujoco
    from scipy.spatial.transform import Rotation
    from .temporal import target_body, world_goal
    times = np.asarray(grid); states = np.asarray(qpos)
    if states.shape != (len(times), robot.model.nq):raise ValueError('METRIC_STATE_GRID_MISMATCH')
    if len(times)<2 or np.any(np.diff(times)<=0):raise ValueError('METRIC_GRID_MUST_INCREASE')
    if not np.isfinite(task_weight) or task_weight<0:raise ValueError('INVALID_TASK_METRIC_WEIGHT')
    if task_objective not in ('center_squared','tolerance_squared_hinge'):raise ValueError('INVALID_METRIC_OBJECTIVE')
    operators = np.einsum('tn,njf->tjf', basis(codec.n, times, codec.duration),
                          codec.N.reshape(codec.n, codec.dof, codec.free_dim) * np.asarray(scale)[None, None, :])
    d = mujoco.MjData(robot.model); local = []; positions = []; rotations = []; collision_matrices=[]; collision_values=[]; collision_rows=0
    if geometry is not None:
        from .convex_guide import ConvexSupportGuide
        if geometry.robot is not robot or not 0<buffer<.02:raise ValueError('INVALID_COLLISION_METRIC_CONTRACT')
        if collision_aggregation not in ('pair_mean','pair_mean_plus_max','pair_mean_plus_max_time_peak'):raise ValueError('INVALID_COLLISION_METRIC_AGGREGATION')
        if not np.isfinite(collision_weight) or collision_weight<0:raise ValueError('INVALID_COLLISION_METRIC_WEIGHT')
        guide=ConvexSupportGuide(geometry)
    for state_index,x in enumerate(states):
        check_budget()
        d.qpos[:] = x
        J, G = instantaneous_tip_jacobians(robot, d)
        local.append(J); positions.append(d.xpos.copy()); rotations.append(d.xmat.reshape(-1,3,3).copy())
        if geometry is not None:
            ids,Jgap,penalties=instantaneous_collision_rows(robot,d,geometry,guide,G,buffer)
            A=(Jgap@operators[state_index])/buffer
            weights=np.full(len(ids),1./len(penalties))
            if collision_aggregation!='pair_mean':weights+=(ids==int(np.argmax(penalties)))
            collision_matrices.append(A.T@(weights[:,None]*A))
            collision_values.append(float(penalties.mean()+(penalties.max() if collision_aggregation!='pair_mean' else 0.)))
            collision_rows+=len(ids)
    local = np.asarray(local); positions = np.asarray(positions); rotations = np.asarray(rotations)
    H = np.zeros((codec.free_dim, codec.free_dim)); rows = 0; peaks = []
    weight = task_weight / max(1, len(task.poses) + len(task.intervals))
    def add(i, target, amount):
        nonlocal H, rows
        J = local[i, target.tip] @ operators[i]
        blocks = []
        if target.position_enabled:blocks.append(J[:3] / target.position_tolerance)
        if target.rotation_enabled:blocks.append(J[3:] / target.rotation_tolerance)
        if blocks:
            A = np.concatenate(blocks, axis=0)
            H += weight * amount * (A.T @ A); rows += len(A)
    for pose in task.poses:
        found = np.flatnonzero(times == pose.time)
        if len(found)!=1:raise ValueError('METRIC_POSE_MISSING_FROM_GRID')
        add(int(found[0]), pose, 1.)
    target_id = target_body(robot)
    for interval in task.intervals:
        ids = np.flatnonzero((times>=interval.start)&(times<=interval.end))
        if len(ids)<2 or times[ids[0]]!=interval.start or times[ids[-1]]!=interval.end:raise ValueError('METRIC_INTERVAL_GRID_INCOMPLETE')
        dt = np.diff(times[ids]); quadrature = np.r_[dt[0]/2, (dt[:-1]+dt[1:])/2, dt[-1]/2] / (interval.end-interval.start)
        errors = []
        for i, amount in zip(ids, quadrature):
            add(i, interval, float(amount))
            P, R = world_goal(interval, positions[i], rotations[i], target_id)
            body = robot.tips[interval.tip]; point = positions[i, body].copy()
            if interval.tip==0:point += rotations[i, body] @ np.array([.0475,0,0])
            values = []
            if interval.position_enabled:values.append(np.linalg.norm(point-P)/interval.position_tolerance)
            if interval.rotation_enabled:values.append(Rotation.from_matrix(R.T@rotations[i,body]).magnitude()/interval.rotation_tolerance)
            errors.append(sum(max(v-task_fraction,0.)**2 if task_objective=='tolerance_squared_hinge' else v*v for v in values))
        worst = int(ids[int(np.argmax(errors))]); add(worst, interval, 1.); peaks.append(float(times[worst]))
    if geometry is not None:
        dt=np.diff(times);quadrature=np.r_[dt[0]/2,(dt[:-1]+dt[1:])/2,dt[-1]/2]/codec.duration
        H+=collision_weight*np.einsum('t,tij->ij',quadrature,np.asarray(collision_matrices))
        if collision_aggregation=='pair_mean_plus_max_time_peak':H+=collision_weight*collision_matrices[int(np.argmax(collision_values))]
    return torch.tensor((H+H.T)/2, dtype=torch.float64), dict(kind='INSTANTANEOUS_MOMENTUM_METRIC_APPROXIMATION',
        full_free_dimension=codec.free_dim, residual_rows=rows, interval_peak_times=peaks,
        native_pose_states=len(states),collision_rows=collision_rows,collision_metric=geometry is not None,
        replaces_path_gradient=False, produces_trajectory=False)

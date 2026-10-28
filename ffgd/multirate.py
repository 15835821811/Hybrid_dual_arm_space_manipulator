"""Optional full-state predictor with bounded inertia-refresh intervals.

Every physical and feedback microstep is retained. Only configuration-dependent
mass and velocity-quadratic bias are held between refreshes; passive damping,
all 67 feedback errors, reference derivatives and external inputs update at
their original microstep. This is a numerical approximation, NOT yet enabled
in the planner. It requires independent native and fine-step calibration.
"""
import torch
import numpy as np
from .physics import tensor


def macro_step(predictor,x,v,u,raw,q,qd,qdd,forces,dt,substeps,start):
    tree=predictor.tree;r=predictor.robot;a=r.dof_ids;b=r.base_dofs
    k=tree.kinematics(x,v);M=tree.mass(x,k)
    h=tree.inverse(x,v,torch.zeros_like(v),k)
    # Preserve the velocity dependence of physical damping at every microstep.
    coriolis=h-tree.damping*v
    response=torch.linalg.inv(M+dt*torch.diag(tree.damping))
    base_response=torch.linalg.inv(M[b[:,None],b])
    kp=tensor(r.spec.position_kp);kd=tensor(r.spec.velocity_kd)
    limits=tensor(r.spec.torque_limits);ids=torch.as_tensor(a,dtype=torch.long)
    xs=[];vs=[];us=[];rs=[]
    for offset in range(len(q)):
        h=coriolis+tree.damping*v
        if (start+offset)%substeps==0:
            aa=tree.B@qdd[offset]
            ab=-base_response@(M[b[:,None],a]@aa+h[b])
            ff=M[a[:,None],b]@ab+M[a[:,None],a]@aa+h[a]
            raw=ff+kp*(tree.B@q[offset]-x[r.qpos_ids])+kd*(tree.B@qd[offset]-v[a])
            u=torch.clamp(raw,-limits,limits)
        force=torch.zeros_like(v).index_copy(0,ids,u)+forces[offset]
        v=v+dt*(response@(force-h));x=predictor.integrate(x,v,dt)
        xs.append(x);vs.append(v);us.append(u);rs.append(raw)
    return tuple(torch.stack(values) for values in (xs,vs,us,rs))


def _step(predictor,*args):
    compiled=getattr(predictor,'compiled_macro_step',None)
    return macro_step(predictor,*args) if compiled is None else compiled(*args)


class MultirateAdjoint(torch.autograd.Function):
    """Recompute one bounded refresh block, retaining all earlier dependence."""
    @staticmethod
    def forward(ctx,x0,v0,q,qd,qdd,forces,predictor,dt,substeps,refresh):
        from .orchestration import check_budget
        x=x0;v=v0;u=torch.zeros(67,dtype=x.dtype);raw=u
        xs=[x[None]];vs=[v[None]];us=[];rs=[]
        for first in range(0,len(q)-1,refresh):
            check_budget();last=min(first+refresh,len(q)-1)
            out=_step(predictor,x,v,u,raw,q[first:last],qd[first:last],qdd[first:last],forces[first:last],dt,substeps,first)
            for rows,values in zip((xs,vs,us,rs),out):rows.append(values)
            x,v,u,raw=(values[-1] for values in out)
        result=tuple(torch.cat(rows) for rows in (xs,vs,us,rs))
        ctx.save_for_backward(*result,q,qd,qdd,forces)
        ctx.predictor=predictor;ctx.dt=dt;ctx.substeps=substeps;ctx.refresh=refresh
        return result

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx,gx,gv,gu,gr):
        from .orchestration import check_budget
        xs,vs,us,rs,q,qd,qdd,forces=ctx.saved_tensors
        adj=[torch.zeros_like(value[0]) for value in (xs,vs,us,rs)]
        gq=torch.zeros_like(q);gqd=torch.zeros_like(qd);gqdd=torch.zeros_like(qdd)
        for first in reversed(range(0,len(us),ctx.refresh)):
            check_budget();last=min(first+ctx.refresh,len(us))
            with torch.enable_grad():
                inputs=[a.detach().requires_grad_() for a in
                        (xs[first],vs[first],us[first-1] if first else torch.zeros_like(us[0]),
                         rs[first-1] if first else torch.zeros_like(rs[0]),q[first:last],qd[first:last],qdd[first:last])]
                out=_step(ctx.predictor,*inputs,forces[first:last],ctx.dt,ctx.substeps,first)
                weights=[gx[first+1:last+1].clone(),gv[first+1:last+1].clone(),gu[first:last].clone(),gr[first:last].clone()]
                for weights_i,carry in zip(weights,adj):weights_i[-1]+=carry
                gradients=torch.autograd.grad(out,inputs,grad_outputs=weights,allow_unused=True)
            gradients=[torch.zeros_like(a) if g is None else g for a,g in zip(inputs,gradients)]
            adj=gradients[:4];gq[first:last]=gradients[4];gqd[first:last]=gradients[5];gqdd[first:last]=gradients[6]
        return adj[0]+gx[0],adj[1]+gv[0],gq,gqd,gqdd,None,None,None,None,None


def rollout(predictor,codec,z,physics_dt=.002,control_substeps=1,duration=None,memory_efficient=True,refresh=5):
    if type(refresh) is not int or not 1<=refresh<=20:raise ValueError('INVALID_INERTIA_REFRESH_BUDGET')
    if type(control_substeps) is not int or control_substeps<1 or physics_dt<=0:raise ValueError('INVALID_MULTIRATE_MICROSTEP')
    stop=codec.duration if duration is None else duration
    if not 0<stop<=codec.duration:raise ValueError('INVALID_MULTIRATE_DURATION')
    steps=int(round(stop/physics_dt))
    if steps<1 or abs(steps*physics_dt-stop)>1e-10:raise ValueError('INVALID_MULTIRATE_GRID')
    times=np.arange(steps+1)*physics_dt
    q=codec.evaluate(z,times);qd=codec.evaluate(z,times,1);qdd=codec.evaluate(z,times,2)
    from .forcing import generalized_forces
    forces=tensor(generalized_forces(predictor.robot,predictor.target_force_pulses,times[:-1]+physics_dt/2))
    xs,vs,us,rs=MultirateAdjoint.apply(predictor.qpos,predictor.velocity,q,qd,qdd,forces,predictor,physics_dt,control_substeps,refresh)
    return dict(time=times,qpos=xs,velocity=vs,applied_torque=us,raw_torque=rs,
                control_updates=(steps+control_substeps-1)//control_substeps,
                integrator='implicit_damping_frozen_inertia',mass_refresh_steps=refresh,
                scope='APPROXIMATE_CONTACT_FREE_FULL_STATE_PREDICTION_NOT_EXECUTION')


def native_prediction(robot,initial,codec,z,dt=.002,refresh=5,substeps=1,target_force_pulses=(),geometry=False):
    """Independent NumPy/MuJoCo-coefficient oracle, not native execution.

    Native M, bias and integration are used. The deliberately frozen M/bias
    approximation is compared separately to actual fine-step MuJoCo dynamics.
    """
    import mujoco
    from .forcing import generalized_forces
    from .orchestration import check_budget
    r=robot;d=r.restore(initial);n=int(round(codec.duration/dt))
    if abs(n*dt-codec.duration)>1e-10:raise ValueError('INVALID_MULTIRATE_NATIVE_GRID')
    times=np.arange(n+1)*dt;q=codec.evaluate(z,times);qd=codec.evaluate(z,times,1);qdd=codec.evaluate(z,times,2)
    forces=generalized_forces(r,target_force_pulses,times[:-1]+dt/2)
    a,b=r.dof_ids,r.base_dofs;D=r.model.dof_damping
    xs=[d.qpos.copy()];vs=[d.qvel.copy()];us=[];raws=[];volumes=[]
    if geometry:
        mujoco.mj_kinematics(r.model,d)
        volumes.append(np.c_[d.geom_xpos,d.geom_xmat*r.model.geom_rbound[:,None]])
    for i in range(n):
        if i%64==0:check_budget()
        if i%refresh==0:
            mujoco.mj_forward(r.model,d);M=r.mass(d);coriolis=d.qfrc_bias-d.qfrc_passive-D*d.qvel
            response=np.linalg.inv(M+dt*np.diag(D));base_response=np.linalg.inv(M[np.ix_(b,b)])
        h=coriolis+D*d.qvel
        if i%substeps==0:
            aa=r.B@qdd[i];ab=-base_response@(M[np.ix_(b,a)]@aa+h[b])
            ff=M[np.ix_(a,b)]@ab+M[np.ix_(a,a)]@aa+h[a]
            raw=ff+r.spec.position_kp*(r.B@q[i]-d.qpos[r.qpos_ids])+r.spec.velocity_kd*(r.B@qd[i]-d.qvel[a])
            applied=np.clip(raw,-r.spec.torque_limits,r.spec.torque_limits)
        force=forces[i].copy();force[a]+=applied
        d.qvel[:]+=dt*(response@(force-h));mujoco.mj_integratePos(r.model,d.qpos,d.qvel,dt);d.time+=dt
        xs.append(d.qpos.copy());vs.append(d.qvel.copy());us.append(applied.copy());raws.append(raw.copy())
        if geometry:
            mujoco.mj_kinematics(r.model,d)
            volumes.append(np.c_[d.geom_xpos,d.geom_xmat*r.model.geom_rbound[:,None]])
    return dict(time=times,qpos=np.asarray(xs),velocity=np.asarray(vs),raw=np.asarray(raws+[raws[-1]]),
                applied_torque=np.asarray(us),volumes=np.asarray(volumes),mass_refresh_steps=refresh,
                scope='NATIVE_COEFFICIENT_NUMERICAL_ORACLE_NOT_EXECUTION')

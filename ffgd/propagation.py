"""Physical-time Lie midpoint propagation with world momentum about fixed O.

Reduced manifold prediction only. Native execution is never projected onto it.
No detach inside the trajectory graph. Target motion/robot external forces must
be explicitly implemented before accepting such initial conditions.
"""
import mujoco
import numpy as np
import torch
from .physics import tensor,quat_matrix,integrate_base


def native_momentum(robot,data):
    """Independent body COM Jacobian/inertia sum, excluding target satellite."""
    m=robot.model
    mujoco.mj_forward(m,data)
    P=np.zeros(3);L=np.zeros(3)
    jp=np.zeros((3,m.nv));jw=np.zeros_like(jp)
    for b in range(1,m.nbody):
        ancestor=b
        while ancestor and ancestor!=robot.base_body: ancestor=int(m.body_parentid[ancestor])
        if ancestor != robot.base_body:continue
        mujoco.mj_jacBodyCom(m,data,jp,jw,b)
        v=jp@data.qvel;w=jw@data.qvel
        R=data.ximat[b].reshape(3,3);I=(R*m.body_inertia[b])@R.T
        p=m.body_mass[b]*v
        P+=p;L+=I@w+np.cross(data.xipos[b],p)
    return np.r_[P,L]


def native_subtree_momentum(robot,data,forward=True):
    """Native subtree COM/angular sum, cross-checked against body Jacobians.

    forward=False requires mj_forward on this exact qpos/qvel immediately before.
    MuJoCo's subtree_angmom is about subtree COM; translate it to fixed world O.
    """
    if forward:mujoco.mj_forward(robot.model,data)
    mujoco.mj_subtreeVel(robot.model,data)
    b=robot.base_body
    P=robot.model.body_subtreemass[b]*data.subtree_linvel[b]
    return np.r_[P,data.subtree_angmom[b]+np.cross(data.subtree_com[b],P)]


class Propagator:
    def __init__(self,tree,initial,manifold_tolerance=1e-10):
        self.tree=tree;self.robot=r=tree.robot
        d=r.restore(initial);m=r.model
        if np.any(d.xfrc_applied) or np.any(d.qfrc_applied):raise ValueError('UNSUPPORTED_EXTERNAL_WRENCH')
        target=m.joint(r.spec.target_free_joint_name).id
        td=int(m.jnt_dofadr[target])
        if np.linalg.norm(d.qvel[td:td+6])>1e-12:raise ValueError('TARGET_MOTION_NOT_IMPLEMENTED')
        q17=r.Bplus@d.qpos[r.qpos_ids];v17=r.Bplus@d.qvel[r.dof_ids]
        self.manifold_error=max(float(np.max(abs(d.qpos[r.qpos_ids]-r.B@q17))),float(np.max(abs(d.qvel[r.dof_ids]-r.B@v17))))
        if self.manifold_error>manifold_tolerance:raise ValueError('INITIAL_MANIFOLD_OUTSIDE_CALIBRATED_DOMAIN')
        self.initial_q=q17;self.initial_v=v17
        self.qpos=tensor(d.qpos);self.base=self.qpos[r.base_qpos]
        self.H=tensor(native_momentum(r,d))
        self.initial_base_velocity=tensor(d.qvel[r.base_dofs])

    def state(self,base,q):
        r=self.robot
        x=self.qpos.clone()
        x[r.base_qpos]=base;x[r.qpos_ids]=self.tree.B@q
        return x

    def velocity(self,base,q,qd):
        r=self.robot;t=self.tree
        x=self.state(base,q);b,a=r.base_dofs,r.dof_ids
        M=t.mass_rows(x,b) if getattr(t,'use_momentum_mass_rows',False) else t.mass(x)[b]
        P,L=self.H[:3],self.H[3:]
        generalized=torch.cat((P,quat_matrix(base[3:]).T@(L-torch.linalg.cross(base[:3],P))))
        va=t.B@qd
        vb=torch.linalg.solve(M[:,b],generalized-M[:,a]@va)
        v=torch.zeros(r.model.nv,dtype=q.dtype);v[b]=vb;v[a]=va
        return x,v

    def bodyframe_mass_rows(self,coordinates):
        """Exact base rows at identity base pose, batched over joint states.

        The robot is a floating subtree; the separate target has zero coupling
        to these rows. No bodies or columns are dropped. The physical momentum
        and base velocities are transformed explicitly in velocity_from_rows.
        This experimental path is off unless the tree flag is set.
        """
        r=self.robot
        if np.any(r.model.dof_armature[r.base_dofs]):
            raise ValueError('BODYFRAME_BATCH_REQUIRES_ZERO_BASE_ARMATURE')
        reference=self.qpos.clone()
        reference[r.base_qpos]=tensor([0.,0.,0.,1.,0.,0.,0.])
        positions=reference[None].expand(len(coordinates),-1).clone()
        positions=positions.index_copy(1,torch.as_tensor(r.qpos_ids,dtype=torch.long),coordinates@self.tree.B.T)
        return torch.vmap(lambda x:self.tree.mass_rows(x,r.base_dofs))(positions)

    def velocity_from_rows(self,base,q,qd,rows):
        r=self.robot;R=quat_matrix(base[3:]);P,L=self.H[:3],self.H[3:]
        target=torch.cat((R.T@P,R.T@(L-torch.linalg.cross(base[:3],P))))
        va=self.tree.B@qd
        body_velocity=torch.linalg.solve(rows[:,r.base_dofs],target-rows[:,r.dof_ids]@va)
        vb=torch.cat((R@body_velocity[:3],body_velocity[3:]))
        v=torch.zeros(r.model.nv,dtype=q.dtype);v[r.base_dofs]=vb;v[r.dof_ids]=va
        return self.state(base,q),v

    def rollout(self,codec,z,intervals=32,check_boundary=True,grid=None):
        times=np.linspace(0,codec.duration,intervals+1) if grid is None else np.asarray(grid,dtype=float)
        if times[0] != 0 or times[-1] != codec.duration or np.any(np.diff(times)<=0):raise ValueError('invalid propagation grid')
        intervals=len(times)-1;mid=(times[:-1]+times[1:])/2
        q=codec.evaluate(z,times);qd=codec.evaluate(z,times,1);qdd=codec.evaluate(z,times,2)
        qm=codec.evaluate(z,mid);vm=codec.evaluate(z,mid,1)
        if check_boundary:
            if np.max(abs(q[0].detach().numpy()-self.initial_q))>1e-10 or np.max(abs(qd[0].detach().numpy()-self.initial_v))>1e-10:
                raise ValueError('REFERENCE_INITIAL_BOUNDARY_MISMATCH')
        base=self.base;xs=[];vs=[]
        mass_rows=None
        if getattr(self.tree,'use_batched_bodyframe_momentum',False):
            mass_rows=self.bodyframe_mass_rows(torch.cat((q,qm)))
        for j in range(intervals+1):
            x,v=self.velocity(base,q[j],qd[j]) if mass_rows is None else self.velocity_from_rows(base,q[j],qd[j],mass_rows[j])
            xs.append(x);vs.append(v)
            if j<intervals:
                dt=times[j+1]-times[j]
                bm=integrate_base(base,v[self.robot.base_dofs],dt/2)
                _,v_mid=self.velocity(bm,qm[j],vm[j]) if mass_rows is None else self.velocity_from_rows(bm,qm[j],vm[j],mass_rows[intervals+1+j])
                base=integrate_base(base,v_mid[self.robot.base_dofs],dt)
        return dict(time=times,q=q,qd=qd,qdd=qdd,qpos=torch.stack(xs),velocity=torch.stack(vs))

    def native_rollout(self,codec,z,intervals,grid=None):
        """Independent same-formula reference using native mass/integration."""
        r=self.robot;m=r.model;d=mujoco.MjData(m);d.qpos[:]=self.qpos.numpy()
        H=self.H.numpy();b,a=r.base_dofs,r.dof_ids
        times=np.linspace(0,codec.duration,intervals+1) if grid is None else np.asarray(grid)
        intervals=len(times)-1
        # Decode once for the complete grid. Re-decoding 323 coordinates at every
        # RK evaluation dominated native dataset validation without changing math.
        q=codec.evaluate(z,times);qd=codec.evaluate(z,times,1)
        mid=(times[:-1]+times[1:])/2
        qm=codec.evaluate(z,mid);qdm=codec.evaluate(z,mid,1)
        def velocity(qvalue,vvalue):
            d.qpos[r.qpos_ids]=r.B@qvalue
            mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);mujoco.mj_crb(m,d)
            M=r.mass(d);base=d.qpos[r.base_qpos];R=d.xmat[r.base_body].reshape(3,3)
            g=np.r_[H[:3],R.T@(H[3:]-np.cross(base[:3],H[:3]))]
            v=np.zeros(m.nv);v[a]=r.B@vvalue
            v[b]=np.linalg.solve(M[np.ix_(b,b)],g-M[np.ix_(b,a)]@v[a])
            return v
        xs=[];vs=[]
        for i in range(intervals+1):
            v=velocity(q[i],qd[i]);xs.append(d.qpos.copy());vs.append(v.copy())
            if i<intervals:
                dt=times[i+1]-times[i]
                origin=d.qpos.copy();mujoco.mj_integratePos(m,d.qpos,v,dt/2)
                vm=velocity(qm[i],qdm[i]);d.qpos[:]=origin
                mujoco.mj_integratePos(m,d.qpos,vm,dt)
        return np.asarray(xs),np.asarray(vs)


class FullStatePredictor:
    """Contact-free closed-loop prediction, opt-in until path calibration passes.

    Predicts the existing controller tracking the supplied reference. Neither
    reference points nor actual states are corrected/projected. Both free
    bodies and all 67 joint states evolve; every state depends on earlier inputs.
    """
    def __init__(self,tree,initial,integrator='midpoint',target_force_pulses=()):
        if integrator not in ('midpoint','implicit_damping'):raise ValueError('UNKNOWN_FULL_STATE_INTEGRATOR')
        self.integrator=integrator
        self.tree=tree;self.robot=r=tree.robot;d=r.restore(initial)
        if np.any(d.xfrc_applied) or np.any(d.qfrc_applied):raise ValueError('UNSUPPORTED_EXTERNAL_WRENCH')
        self.qpos=tensor(d.qpos);self.velocity=tensor(d.qvel)
        self.initial_state=initial
        self.target_force_pulses=tuple(target_force_pulses)

    def integrate(self,x,v,dt):
        r=self.robot;y=x.clone()
        y[r.qpos_ids]=x[r.qpos_ids]+dt*v[r.dof_ids]
        for j,kind in enumerate(r.model.jnt_type):
            if kind==int(mujoco.mjtJoint.mjJNT_FREE):
                qa=int(r.model.jnt_qposadr[j]);va=int(r.model.jnt_dofadr[j])
                y[qa:qa+7]=integrate_base(x[qa:qa+7],v[va:va+6],dt)
        return y

    def rollout(self,codec,z,physics_dt=.0005,control_substeps=4,duration=None,memory_efficient=False):
        if type(control_substeps) is not int or control_substeps<1:raise ValueError('INVALID_CONTROL_SUBSTEPS')
        stop=codec.duration if duration is None else duration
        steps=int(round(stop/physics_dt))
        if physics_dt<=0 or stop<=0 or stop>codec.duration or abs(steps*physics_dt-stop)>1e-10:raise ValueError('INVALID_PREDICTION_GRID')
        times=np.arange(steps+1)*physics_dt
        q=codec.evaluate(z,times);qd=codec.evaluate(z,times,1);qdd=codec.evaluate(z,times,2)
        from .forcing import generalized_forces
        forces=tensor(generalized_forces(self.robot,self.target_force_pulses,times[:-1]+physics_dt/2))
        if memory_efficient:
            xs,vs,torques,raws=FullStateAdjoint.apply(self.qpos,self.velocity,q,qd,qdd,forces,self,physics_dt,control_substeps)
            return dict(time=times,qpos=xs,velocity=vs,applied_torque=torques,raw_torque=raws,
                        control_updates=(steps+control_substeps-1)//control_substeps,
                        integrator=self.integrator,scope='CONTACT_FREE_FULL_STATE_PREDICTION_NOT_NATIVE_EXECUTION')
        x=self.qpos;v=self.velocity;xs=[x];vs=[v];torques=[];raws=[]
        for i in range(steps):
            if i%16==0:
                from .orchestration import check_budget
                check_budget()
            if i==0:applied=torch.zeros(67,dtype=x.dtype);raw=applied
            x,v,applied,raw=self.step(x,v,applied,raw,q[i],qd[i],qdd[i],physics_dt,i%control_substeps==0,forces[i])
            xs.append(x);vs.append(v);torques.append(applied);raws.append(raw)
        return dict(time=times,qpos=torch.stack(xs),velocity=torch.stack(vs),
                    applied_torque=torch.stack(torques),raw_torque=torch.stack(raws),
                    control_updates=(steps+control_substeps-1)//control_substeps,
                    integrator=self.integrator,scope='CONTACT_FREE_FULL_STATE_PREDICTION_NOT_NATIVE_EXECUTION')


    def step(self,x,v,applied,raw,q,qd,qdd,dt,update,external=None):
        if external is None:external=torch.zeros_like(v)
        if self.integrator=='implicit_damping':
            if update:applied,raw,M,h=self.tree.tracking_command(x,v,q,qd,qdd)
            else:
                k=self.tree.kinematics(x,v);M=self.tree.mass(x,k)
                h=self.tree.inverse(x,v,torch.zeros_like(v),k)
            force=torch.zeros_like(v).index_copy(0,torch.as_tensor(self.robot.dof_ids,dtype=torch.long),applied)
            dv=dt*torch.linalg.solve(M+dt*torch.diag(self.tree.damping),force+external-h)
            vn=v+dv
            return self.integrate(x,vn,dt),vn,applied,raw
        if update:applied,raw,_,_=self.tree.tracking_command(x,v,q,qd,qdd)
        if self.target_force_pulses:raise ValueError('FORCED_PREDICTION_REQUIRES_IMPLICIT_DAMPING')
        a=self.tree.full_state_acceleration(x,v,applied)
        xm=self.integrate(x,v,dt/2);vm=v+a*(dt/2)
        am=self.tree.full_state_acceleration(xm,vm,applied)
        return self.integrate(x,vm,dt),v+dt*am,applied,raw


class FullStateAdjoint(torch.autograd.Function):
    """Exact first-order discrete adjoint of the implemented midpoint recurrence.

    Save numeric states, not the full rigid-body autograd graph at every time.
    Backward reconstructs one step, retaining held torque/raw demand as recurrent
    state. This is not a continuous adjoint, detached-prefix approximation or FD.
    """
    @staticmethod
    def forward(ctx,x0,v0,q,qd,qdd,forces,predictor,dt,substeps):
        x=x0;v=v0;u=torch.zeros(67,dtype=x.dtype);raw=u
        xs=[x];vs=[v];us=[];raws=[]
        for i in range(len(q)-1):
            if i%16==0:
                from .orchestration import check_budget
                check_budget()
            x,v,u,raw=predictor.step(x,v,u,raw,q[i],qd[i],qdd[i],dt,i%substeps==0,forces[i])
            xs.append(x);vs.append(v);us.append(u);raws.append(raw)
        result=tuple(torch.stack(a) for a in (xs,vs,us,raws))
        ctx.save_for_backward(*result,q,qd,qdd,forces)
        ctx.predictor=predictor;ctx.dt=dt;ctx.substeps=substeps
        return result

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx,gx,gv,gu,gr):
        xs,vs,us,raws,q,qd,qdd,forces=ctx.saved_tensors
        ax=torch.zeros_like(xs[0]);av=torch.zeros_like(vs[0])
        au=torch.zeros_like(us[0]);ar=torch.zeros_like(raws[0])
        gq=torch.zeros_like(q);gqd=torch.zeros_like(qd);gqdd=torch.zeros_like(qdd)
        for i in range(len(us)-1,-1,-1):
            if i%16==0:
                from .orchestration import check_budget
                check_budget()
            with torch.enable_grad():
                values=[xs[i],vs[i],us[i-1] if i else torch.zeros_like(au),raws[i-1] if i else torch.zeros_like(ar),q[i],qd[i],qdd[i]]
                inputs=[value.detach().requires_grad_(True) for value in values]
                out=ctx.predictor.step(*inputs,ctx.dt,i%ctx.substeps==0,forces[i])
                gradients=torch.autograd.grad(out,inputs,(ax+gx[i+1],av+gv[i+1],au+gu[i],ar+gr[i]),allow_unused=True)
            ax,av,au,ar,dq,dqd,dqdd=[torch.zeros_like(value) if grad is None else grad for value,grad in zip(values,gradients)]
            gq[i]=dq;gqd[i]=dqd;gqdd[i]=dqdd
        return ax+gx[0],av+gv[0],gq,gqd,gqdd,None,None,None,None


class FullStatePropagation:
    """Whole-state guide adapter; independent native integration is separate."""
    prediction_mode='full_state'
    def __init__(self,tree,initial,physics_dt=.002,target_force_pulses=()):
        self.tree=tree;self.robot=r=tree.robot;self.initial=initial
        d=r.restore(initial)
        self.qpos=tensor(d.qpos);self.initial_q=r.Bplus@d.qpos[r.qpos_ids]
        self.initial_v=r.Bplus@d.qvel[r.dof_ids];self.H=tensor(native_momentum(r,d))
        if physics_dt not in (.0005,.001,.002,.004,.01):raise ValueError('UNSUPPORTED_DECLARED_PREDICTION_TIMESTEP')
        self.dt=physics_dt
        self.predicted_control_period=max(.002,physics_dt)
        self.target_force_pulses=tuple(target_force_pulses)
        self.predictor=FullStatePredictor(tree,initial,integrator='implicit_damping',target_force_pulses=self.target_force_pulses)

    def rollout(self,codec,z,intervals=32,check_boundary=True,grid=None):
        times=np.linspace(0,codec.duration,intervals+1) if grid is None else np.asarray(grid)
        q=codec.evaluate(z,times);qd=codec.evaluate(z,times,1);qdd=codec.evaluate(z,times,2)
        reference=getattr(codec,'reference_start',np.stack((self.initial_q,self.initial_v,np.zeros(17))))
        if check_boundary and (np.max(abs(q[0].detach().numpy()-reference[0]))>1e-10 or np.max(abs(qd[0].detach().numpy()-reference[1]))>1e-10):raise ValueError('REFERENCE_INITIAL_BOUNDARY_MISMATCH')
        substeps=int(round(self.predicted_control_period/self.dt))
        full=self.predictor.rollout(codec,z,physics_dt=self.dt,control_substeps=substeps,memory_efficient=True)
        # Geometry/kinematics at off-grid spline knots use linear state sampling;
        # task times remain unchanged. Quaternion normalization occurs in FK.
        coordinate=np.clip(times/self.dt,0,len(full['time'])-1)
        left=np.floor(coordinate).astype(int);right=np.minimum(left+1,len(full['time'])-1)
        alpha=tensor(coordinate-left)[:,None]
        x=full['qpos'][left]*(1-alpha)+full['qpos'][right]*alpha
        v=full['velocity'][left]*(1-alpha)+full['velocity'][right]*alpha
        control_indices=np.minimum(left,len(full['raw_torque'])-1)
        raw=full['raw_torque'][control_indices];applied=full['applied_torque'][control_indices]
        from .forcing import generalized_forces
        forces=tensor(generalized_forces(self.robot,self.target_force_pulses,times))
        acceleration=torch.stack([self.tree.full_state_acceleration(xx,vv,uu,ff) for xx,vv,uu,ff in zip(x,v,applied,forces)])
        r=self.robot
        return dict(time=times,q=q,qd=qd,qdd=qdd,qpos=x,velocity=v,
                    actual_q67=x[:,r.qpos_ids],actual_v67=v[:,r.dof_ids],actual_a67=acceleration[:,r.dof_ids],
                    feedback_raw_torque=raw,prediction_steps=len(full['time'])-1)

    def native_rollout(self,codec,z,intervals=32,grid=None):
        from .full_state_validation import native_prediction
        full=native_prediction(self.robot,self.initial,codec,z,self.dt,geometry=False,target_force_pulses=self.target_force_pulses)
        self.last_native_prediction=full
        grid=np.linspace(0,codec.duration,intervals+1) if grid is None else np.asarray(grid)
        coordinate=np.clip(grid/self.dt,0,len(full['qpos'])-1);left=np.floor(coordinate).astype(int);right=np.minimum(left+1,len(full['qpos'])-1);a=(coordinate-left)[:,None]
        return full['qpos'][left]*(1-a)+full['qpos'][right]*a,full['velocity'][left]*(1-a)+full['velocity'][right]*a

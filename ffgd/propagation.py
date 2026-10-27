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
        x=self.state(base,q);M=t.mass(x);b,a=r.base_dofs,r.dof_ids
        P,L=self.H[:3],self.H[3:]
        generalized=torch.cat((P,quat_matrix(base[3:]).T@(L-torch.linalg.cross(base[:3],P))))
        va=t.B@qd
        vb=torch.linalg.solve(M[b[:,None],b],generalized-M[b[:,None],a]@va)
        v=torch.zeros(r.model.nv,dtype=q.dtype);v[b]=vb;v[a]=va
        return x,v

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
        for j in range(intervals+1):
            x,v=self.velocity(base,q[j],qd[j]);xs.append(x);vs.append(v)
            if j<intervals:
                dt=times[j+1]-times[j]
                bm=integrate_base(base,v[self.robot.base_dofs],dt/2)
                _,v_mid=self.velocity(bm,qm[j],vm[j])
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

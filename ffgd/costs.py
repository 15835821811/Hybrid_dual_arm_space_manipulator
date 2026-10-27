"""Five dimensionless multi-space energies evaluated on the same whole path."""
import numpy as np
import torch
from .physics import tensor
from .trajectory import knots
from .convex_guide import ConvexSupportGuide


def rotation_angle(R):
    vee=torch.stack((R[2,1]-R[1,2],R[0,2]-R[2,0],R[1,0]-R[0,1]))/2
    sine=torch.linalg.vector_norm(vee);cosine=((torch.trace(R)-1)/2).clamp(-1,1)
    if float(cosine.detach())<0 and float(sine.detach())<1e-8:raise ValueError('SO3_PI_NONDIFFERENTIABLE')
    return torch.atan2(sine,cosine)


def hinge(x):return torch.relu(x).square()


def rotation_log(R,angle=None):
    theta=rotation_angle(R) if angle is None else angle
    vee=torch.stack((R[2,1]-R[1,2],R[0,2]-R[2,0],R[1,0]-R[0,1]))/2
    sine=torch.linalg.vector_norm(vee);cosine=(torch.trace(R)-1)/2
    if float(cosine.detach())<-.99:
        # Near pi the antisymmetric part is tiny: normalizing it loses axis
        # derivatives to cancellation. Recover the same signed axis from the
        # symmetric part using its largest diagonal component instead.
        index=int(torch.argmax(torch.diag(R)).detach())
        component=torch.sqrt((R[index,index]-cosine)/(1-cosine))
        axis=torch.stack([component if j==index else (R[index,j]+R[j,index])/(2*(1-cosine)*component) for j in range(3)])
        axis=axis/torch.linalg.vector_norm(axis)
        if float((axis@vee).detach())<0:axis=-axis
        return theta*axis
    ratio=torch.where((sine<1e-5)&(cosine>0),1+sine.square()/6+3*sine**4/40,theta/sine.clamp_min(1e-12))
    return ratio*vee


class MultiSpaceEnergy:
    def __init__(self,propagator,codec,geometry,task,intervals=32,buffer=.002,torque_fraction=.8,collect_task_residual=False,collision_aggregation='pair_mean',joint_peak_penalty=False):
        if codec.duration!=task.duration:raise ValueError('time contract mismatch')
        self.p,self.codec,self.geometry,self.task=propagator,codec,geometry,task
        self.grid=np.unique(np.r_[np.linspace(0,task.duration,intervals+1),knots(codec.n)*task.duration,[x.time for x in task.poses]])
        self.buffer,self.torque_fraction=buffer,torque_fraction
        self.weights=dict(task=1.,collision=1.,joint=1.,actuator=1.,smoothness=1e-3)
        self.collision_guide=ConvexSupportGuide(geometry)
        self.collect_task_residual=collect_task_residual;self.last_task_residual=None
        if collision_aggregation not in ('pair_mean','pair_mean_plus_max'):raise ValueError('unknown collision aggregation')
        self.collision_aggregation=collision_aggregation
        self.joint_peak_penalty=joint_peak_penalty

    def evaluate(self,z,return_path=False):
        out=self.p.rollout(self.codec,z,grid=self.grid);tree=self.p.tree;r=self.p.robot
        task=z.sum()*0;collision=[];actuator=[];smooth=[];raw=[];minima=[]
        support_pairs=0
        task_residuals=[]
        b,a=r.base_dofs,r.dof_ids
        for i,t in enumerate(self.grid):
            x,v=out['qpos'][i],out['velocity'][i]
            k=tree.kinematics(x,v)
            gaps=self.collision_guide.gaps(k);minima.append(gaps.min())
            support_pairs+=self.collision_guide.last_query['support_pairs']
            penalties=hinge((self.buffer-gaps)/self.buffer)
            # Time-normalized mean plus worst-pair pressure cannot dilute a
            # local violation by adding thousands of safe geometry pairs.
            collision.append(penalties.mean()+(penalties.max() if self.collision_aggregation=='pair_mean_plus_max' else 0.))
            M=tree.mass(x,k);h=tree.inverse(x,v,torch.zeros_like(v),k)
            aa=tree.B@out['qdd'][i]
            ab=-torch.linalg.solve(M[b[:,None],b],M[b[:,None],a]@aa+h[b])
            tau=M[a[:,None],b]@ab+M[a[:,None],a]@aa+h[a];raw.append(tau)
            actuator.append(hinge(tau.abs()/(tensor(r.spec.torque_limits)*self.torque_fraction)-1).mean())
            smooth.append(out['qdd'][i].square().mean())
            for target in self.task.poses:
                if target.time != t:continue
                body=r.tips[target.tip];pos=k['position'][body]
                if target.tip==0:pos=pos+k['rotation'][body]@tensor([.0475,0,0])
                if target.position_enabled:
                    residual=(pos-tensor(target.position))/target.position_tolerance;task=task+residual.square().sum()
                    if self.collect_task_residual:task_residuals.append(residual)
                if target.rotation_enabled:
                    R=tensor(target.rotation).T@k['rotation'][body];angle=rotation_angle(R)
                    task=task+(angle/target.rotation_tolerance)**2
                    if self.collect_task_residual:task_residuals.append(rotation_log(R,angle)/target.rotation_tolerance)
        q,qd,qdd=out['q'],out['qd'],out['qdd']
        joint=hinge((tensor(r.lower)-q)/(tensor(r.upper-r.lower))).mean(-1)+hinge((q-tensor(r.upper))/tensor(r.upper-r.lower)).mean(-1)
        joint=joint+hinge(qd.abs()/tensor(r.spec.planner_velocity_limits)-1).mean(-1)+hinge(qdd.abs()/tensor(r.spec.planner_acceleration_limits)-1).mean(-1)
        limited=r.model.jnt_limited[r.joints].astype(bool);limits=r.model.jnt_range[r.joints][limited]
        qa=q@tree.B.T
        if limited.any():
            span=tensor(limits[:,1]-limits[:,0]);actual=qa[:,limited]
            joint=joint+hinge((tensor(limits[:,0])-actual)/span).mean(-1)+hinge((actual-tensor(limits[:,1]))/span).mean(-1)
        time=tensor(self.grid)
        def integral(values):return torch.trapezoid(torch.stack(values) if isinstance(values,list) else values,time)/self.task.duration
        terms=dict(task=task/len(self.task.poses),collision=integral(collision),joint=integral(joint),actuator=integral(actuator),smoothness=integral(smooth))
        if self.joint_peak_penalty:
            # Keep the original time average, but do not let one violated
            # joint/knot vanish among safe coordinates or a long duration.
            peaks=[hinge((tensor(r.lower)-q)/tensor(r.upper-r.lower)).max(),
                   hinge((q-tensor(r.upper))/tensor(r.upper-r.lower)).max(),
                   hinge(qd.abs()/tensor(r.spec.planner_velocity_limits)-1).max(),
                   hinge(qdd.abs()/tensor(r.spec.planner_acceleration_limits)-1).max()]
            if limited.any():peaks.extend([hinge((tensor(limits[:,0])-actual)/span).max(),hinge((actual-tensor(limits[:,1]))/span).max()])
            terms['joint']=terms['joint']+torch.stack(peaks).max()
        terms['total']=sum(self.weights[k]*v for k,v in terms.items())
        self.last_task_residual=(torch.cat(task_residuals)*np.sqrt(self.weights['task']/len(self.task.poses))) if task_residuals else None
        self.last_diagnostics=dict(grid_points=len(self.grid),all_eligible_pairs=len(self.geometry.pairs),
                                   collision_aggregation=self.collision_aggregation,
                                   joint_peak_penalty=self.joint_peak_penalty,
                                   support_pair_evaluations=support_pairs,broadphase_pair_evaluations=len(self.grid)*len(self.geometry.pairs),
                                   mass_evaluations=3*len(self.grid)-1,
                                   minimum_proxy_gap=float(torch.stack(minima).min().detach()),
                                   maximum_raw_torque_ratio=float((torch.stack(raw).abs()/tensor(r.spec.torque_limits)).max().detach()))
        if return_path:return terms,dict(**out,raw_torque=torch.stack(raw),proxy_minimum=torch.stack(minima))
        return terms

    def native_evaluate(self,z):
        """Independent native scalar oracle for finite differences/verification.

        Never used as a finite-difference gradient backend in the planner.
        """
        import mujoco
        from scipy.spatial.transform import Rotation
        p=self.p;r=p.robot;m=r.model;g=self.geometry
        xs,vs=p.native_rollout(self.codec,z,len(self.grid)-1,grid=self.grid)
        q=self.codec.evaluate(z,self.grid);qd=self.codec.evaluate(z,self.grid,1);qdd=self.codec.evaluate(z,self.grid,2)
        task=0.;coll=[];act=[];a,b=r.dof_ids,r.base_dofs;d=mujoco.MjData(m)
        def h(x):return np.maximum(x,0)**2
        for i,t in enumerate(self.grid):
            d.qpos[:]=xs[i];d.qvel[:]=vs[i];mujoco.mj_forward(m,d)
            M=r.mass(d);bias=d.qfrc_bias-d.qfrc_passive;aa=r.B@qdd[i]
            ab=-np.linalg.solve(M[np.ix_(b,b)],M[np.ix_(b,a)]@aa+bias[b])
            tau=M[np.ix_(a,b)]@ab+M[np.ix_(a,a)]@aa+bias[a]
            act.append(h(abs(tau)/(r.spec.torque_limits*self.torque_fraction)-1).mean())
            distances=self.collision_guide.numpy_gaps(d.geom_xpos[g.ids],d.geom_xmat[g.ids].reshape(-1,3,3))
            penalties=h((self.buffer-distances)/self.buffer)
            coll.append(penalties.mean()+(penalties.max() if self.collision_aggregation=='pair_mean_plus_max' else 0.))
            for target in self.task.poses:
                if t!=target.time:continue
                body=r.tips[target.tip];R=d.xmat[body].reshape(3,3);pos=d.xpos[body].copy()
                if target.tip==0:pos+=R@np.array([.0475,0,0])
                if target.position_enabled:task+=np.sum(((pos-target.position)/target.position_tolerance)**2)
                if target.rotation_enabled:task+=(Rotation.from_matrix(target.rotation.T@R).magnitude()/target.rotation_tolerance)**2
        joint=h((r.lower-q)/(r.upper-r.lower)).mean(-1)+h((q-r.upper)/(r.upper-r.lower)).mean(-1)
        joint+=h(abs(qd)/r.spec.planner_velocity_limits-1).mean(-1)+h(abs(qdd)/r.spec.planner_acceleration_limits-1).mean(-1)
        limited=m.jnt_limited[r.joints].astype(bool);limits=m.jnt_range[r.joints][limited];qa=q@r.B.T
        if limited.any():
            span=limits[:,1]-limits[:,0];actual=qa[:,limited]
            joint+=h((limits[:,0]-actual)/span).mean(-1)+h((actual-limits[:,1])/span).mean(-1)
        def integral(v):return float(np.trapz(v,self.grid)/self.task.duration)
        terms=dict(task=task/len(self.task.poses),collision=integral(coll),joint=integral(joint),actuator=integral(act),smoothness=integral((qdd**2).mean(-1)))
        if self.joint_peak_penalty:
            peaks=[np.max(h((r.lower-q)/(r.upper-r.lower))),np.max(h((q-r.upper)/(r.upper-r.lower))),
                   np.max(h(abs(qd)/r.spec.planner_velocity_limits-1)),np.max(h(abs(qdd)/r.spec.planner_acceleration_limits-1))]
            if limited.any():peaks.extend([np.max(h((limits[:,0]-actual)/span)),np.max(h((actual-limits[:,1])/span))])
            terms['joint']+=max(peaks)
        terms['total']=sum(self.weights[k]*v for k,v in terms.items())
        return terms

    def native_task_residual(self,z):
        """Independent native FK/SciPy SO(3) oracle for Jacobian calibration."""
        import mujoco
        from scipy.spatial.transform import Rotation
        r=self.p.robot;d=mujoco.MjData(r.model)
        xs,vs=self.p.native_rollout(self.codec,z,len(self.grid)-1,grid=self.grid)
        residual=[]
        for i,t in enumerate(self.grid):
            for target in self.task.poses:
                if target.time!=t:continue
                d.qpos[:]=xs[i];mujoco.mj_kinematics(r.model,d)
                body=r.tips[target.tip];R=d.xmat[body].reshape(3,3);pos=d.xpos[body].copy()
                if target.tip==0:pos+=R@np.array([.0475,0,0])
                if target.position_enabled:residual.extend((pos-target.position)/target.position_tolerance)
                if target.rotation_enabled:residual.extend(Rotation.from_matrix(target.rotation.T@R).as_rotvec()/target.rotation_tolerance)
        return np.asarray(residual)*np.sqrt(self.weights['task']/len(self.task.poses))

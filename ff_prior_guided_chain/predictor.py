"""Planning scratch states only; physical-time transport of inherited momentum.

No mj_step, reset of an actual object, or zeroing of inherited native arm modes.
The unrepresented arm position/velocity residual is propagated ballistically;
its initial value and maximum magnitude are reported, and execution still uses
all 67 actuator states and the original manifold gate.
"""
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from tracking_diffusion.environment import free_slices
from ff_prior_satellite_origin.force import analytic_motion


def world_momentum(base_generalized,position,rotation):
    p=np.asarray(base_generalized)[:3]
    return p.copy(),rotation@np.asarray(base_generalized)[3:]+np.cross(position,p)


def generalized_momentum(p,h,position,rotation):
    return np.r_[p,rotation.T@(h-np.cross(position,p))]


class Predictor:
    def __init__(self,robot,initial,phase,budget):
        self.r,self.initial,self.phase,self.budget=robot,initial,phase,budget
        d=initial.reset(robot);robot.forward(d)
        self.qpos0=d.qpos.copy();self.qvel0=d.qvel.copy()
        self.q0=robot.Bplus@d.qpos[robot.qpos_ids]
        self.v0=robot.Bplus@d.qvel[robot.dof_ids]
        self.resq=d.qpos[robot.qpos_ids]-robot.B@self.q0
        self.resv=d.qvel[robot.dof_ids]-robot.B@self.v0
        b=robot.base_dofs;bp=robot.base_qpos
        R=Rotation.from_quat(np.roll(d.qpos[bp[3:]],-1)).as_matrix()
        self.p,self.h=world_momentum((robot.mass(d)@d.qvel)[b],d.qpos[bp[:3]],R)
        self.tp,self.tv=free_slices(robot.model,robot.spec.target_free_joint_name)
        self.target=robot.id(mujoco.mjtObj.mjOBJ_BODY,'target_satellite')
        self.mass_target=float(robot.model.body_mass[self.target])
        if np.any(d.xfrc_applied) or np.any(d.qfrc_applied):raise ValueError('PHASE_INITIAL_EXTERNAL_WRENCH_NOT_ZERO')

    def run(self,ref,times):
        r=self.r;m=r.model;b=r.base_dofs;a=r.dof_ids;bp=r.base_qpos
        times=np.asarray(times,dtype=float)
        if times[0]!=0 or np.any(np.diff(times)<=0):raise ValueError('PLANNING_GRID')
        self.budget.propagation(len(times)-1);r.calls['coupled']+=1
        d=self.initial.reset(r);residual=0.;qnorm=0.
        records={k:[] for k in ('qpos','velocity','ee','base','geom','geom_rotation','raw_torque','momentum','exchange')}
        bb=np.ix_(b,b);ba=np.ix_(b,a)
        def set_target(t):
            v=self.qvel0[self.tv].copy()
            pos=self.qpos0[self.tp.start:self.tp.start+3]+v[:3]*t
            rot=Rotation.from_quat(np.roll(self.qpos0[self.tp.start+3:self.tp.stop],-1))*Rotation.from_rotvec(v[3:]*t)
            if self.phase=='C':
                z,dz=analytic_motion(t,self.mass_target);pos[2]+=z;v[2]+=dz
            d.qpos[self.tp]=np.r_[pos,np.roll(rot.as_quat(),1)]
            return v
        def at(t):
            nonlocal residual,qnorm
            d.qpos[r.qpos_ids]=r.B@ref.evaluate(t)+self.resq+t*self.resv
            vt=set_target(t);r.kinematics(d,mass=True);mass=r.mass(d)
            v=self.qvel0.copy();v[a]=r.B@ref.evaluate(t,1)+self.resv;v[self.tv]=vt;v[b]=0.
            R=Rotation.from_quat(np.roll(d.qpos[bp[3:]],-1)).as_matrix()
            target=generalized_momentum(self.p,self.h,d.qpos[bp[:3]],R)
            v[b]=np.linalg.solve(mass[bb],target-(mass@v)[b])
            residual=max(residual,float(np.max(abs((mass@v)[b]-target))))
            qnorm=max(qnorm,abs(float(np.linalg.norm(d.qpos[bp[3:]]))-1))
            return v
        for i,t in enumerate(times):
            self.budget.check();v=at(t);d.qvel[:]=v;r.forward(d);mass=r.mass(d)
            force=d.qfrc_bias-d.qfrc_passive;acc=r.B@ref.evaluate(t,2)
            ab=-np.linalg.solve(mass[bb],mass[ba]@acc+force[b])
            raw=mass[np.ix_(a,b)]@ab+mass[np.ix_(a,a)]@acc+force[a]
            vals=dict(qpos=d.qpos,velocity=v,ee=r.ee(d),base=d.qpos[bp],geom=d.geom_xpos,
                geom_rotation=d.geom_xmat,raw_torque=raw,momentum=(mass@v)[b],exchange=mass[ba]@v[a])
            for k,val in vals.items():records[k].append(val.copy())
            if i+1<len(times):
                dt=times[i+1]-t;origin=d.qpos.copy()
                mujoco.mj_integratePos(m,d.qpos,v,dt/2)
                vm=at(t+dt/2);d.qpos[:]=origin
                mujoco.mj_integratePos(m,d.qpos,vm,dt)
        self.budget.complete('coupled');self.budget.complete('intervals',len(times)-1)
        result={k:np.asarray(v) for k,v in records.items()}
        result.update(time=times,numerical=dict(solve_residual=residual,quaternion_error=qnorm,
            inherited_world_linear_momentum=self.p,inherited_world_angular_momentum=self.h,
            initial_native_residual_q=self.resq,initial_native_residual_v=self.resv,
            initial_qpos_error=float(np.max(abs(result['qpos'][0]-self.qpos0))),
            initial_qvel_error=float(np.max(abs(result['velocity'][0]-self.qvel0))),
            method='physical_time_midpoint_inherited_world_momentum',native_steps=0))
        if residual>1e-10 or qnorm>1e-12:raise ValueError('MODEL_NUMERICAL_GATE')
        if result['numerical']['initial_qpos_error']>1e-10 or result['numerical']['initial_qvel_error']>1e-10:
            raise ValueError('MODEL_INITIAL_FULL_STATE_MISMATCH')
        return result

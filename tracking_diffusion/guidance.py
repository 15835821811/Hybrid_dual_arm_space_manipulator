"""Late DDIM guidance through the floating-base local kinematic model.

Gradients are analytic in spline control points. Native MuJoCo supplies the
measured-state Jacobians, including base coupling and collision witnesses.
The short-horizon linearization is an approximation, not a dynamics certificate.
No teacher trajectory or teacher control command is read during generation.
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
import torch

from .spline import SplineChunk, basis_spline, knots, VELOCITY_FACTOR


class ConstraintGuidance:
    def __init__(self,env,horizon,mean,scale,device,project=True,diagnostic=False):
        self.env,self.h,self.mean,self.scale,self.device=env,horizon,mean,scale,device
        self.q,self.dq=env.state()
        self.history=[]
        self.project=project
        self.diagnostic=diagnostic
        self.stages={}
        g,drift=env.generalized_map()
        j,offset=env.task_jacobian(g,drift)
        env.clearance(g,drift)
        collision=env.collision_linearization
        times=np.linspace(0.,horizon,11)
        basis=basis_spline(8)(times/horizon)
        identity=SplineChunk(np.eye(8),horizon)
        v,a=identity.derivative_controls()
        self.derivatives=np.vstack([np.eye(8),v,a])
        self.fixed=np.stack([self.q,self.q+horizon*knots(8)[4]*self.dq/3.])
        base=SplineChunk.decode(mean,self.q,self.dq,horizon).controls
        mapping=np.zeros((8*17,6*17))
        mapping[2*17:]=np.diag((horizon*scale).ravel())
        positions=basis@base
        pmap=np.kron(basis,np.eye(17))@mapping
        task_scale=np.tile(np.repeat([.005,.02,.005,.02],3),len(times))
        self.task_matrix=(np.kron(np.eye(len(times)),j)@pmap)/task_scale[:,None]
        target=np.array([env.task_error(float(env.data.time)+t) for t in times])
        self.task_offset=(((positions-self.q)@j.T+times[:,None]*offset-target).ravel()/task_scale)
        self.systems=[]
        # Smooth barrier penalties are differentiated BEFORE final filtering.
        for matrix,lo,hi,unit,weight,name in (
            (np.eye(8),env.spec.planner_lower+.04,env.spec.planner_upper-.04,np.ones(17),4.,"position"),
            (v,-.70*env.spec.planner_velocity_limits,.70*env.spec.planner_velocity_limits,
             env.spec.planner_velocity_limits,2.,"velocity"),
            (a,-.80*env.spec.planner_acceleration_limits,.80*env.spec.planner_acceleration_limits,
             env.spec.planner_acceleration_limits,2.,"acceleration")):
            n=len(matrix)
            units=np.tile(unit,n)/np.sqrt(weight)
            self.systems.append((np.kron(matrix,np.eye(17))@mapping/units[:,None],
                (matrix@base).ravel()/units,np.tile(lo,n)/units,np.tile(hi,n)/units,name))
        if len(collision["margin"]):
            cm=collision["jacobian"]
            units=.005/np.sqrt(4.)
            self.systems.append((np.kron(np.eye(len(times)),cm)@pmap/units,
                (collision["margin"][None]+times[:,None]*collision["drift"]+
                 (positions-self.q)@cm.T).ravel()/units,
                np.full(len(times)*len(cm),.001/units),
                np.full(len(times)*len(cm),np.inf),"collision"))
        # A constant positive-definite upper bound on the local Hessian acts as
        # a gradient preconditioner. It does not call inverse-kinematic teaching.
        hessian=self.task_matrix.T@self.task_matrix+.05*np.eye(102)
        for matrix,_,_,_,_ in self.systems:
            hessian+=matrix.T@matrix
        self.factor=cho_factor(hessian,lower=True,check_finite=False)

    def energy_gradient(self,z):
        r=z@self.task_matrix.T+self.task_offset
        energy=.5*np.sum(r*r,axis=1)
        grad=r@self.task_matrix
        for matrix,offset,lo,hi,_ in self.systems:
            values=z@matrix.T+offset
            residual=values-np.clip(values,lo,hi)
            energy+=.5*np.sum(residual*residual,axis=1)
            grad+=residual@matrix
        return energy,grad

    def project_joint_bounds(self,free):
        controls=SplineChunk.decode(free,self.q,self.dq,self.h).controls.copy()
        m=self.derivatives[:,2:]
        offset=self.derivatives[:,:2]@self.fixed
        # This is a small nearest-control-point projection after the final
        # gradient step, explicitly recorded separately from learned generation.
        for j in range(17):
            spec=self.env.spec
            lo=np.r_[np.full(8,spec.planner_lower[j]+.040001),
                np.full(7,-(VELOCITY_FACTOR-.0001)*spec.planner_velocity_limits[j]),
                np.full(6,-.9998*spec.planner_acceleration_limits[j])]-offset[:,j]
            hi=np.r_[np.full(8,spec.planner_upper[j]-.040001),
                np.full(7,(VELOCITY_FACTOR-.0001)*spec.planner_velocity_limits[j]),
                np.full(6,.9998*spec.planner_acceleration_limits[j])]-offset[:,j]
            original=controls[2:,j].copy()
            if np.all(m@original>=lo) and np.all(m@original<=hi):
                continue
            result=minimize(lambda z:.5*np.sum((z-original)**2),original,
                jac=lambda z:z-original,
                constraints=({"type":"ineq","fun":lambda z:np.r_[m@z-lo,hi-m@z],
                              "jac":lambda z:np.vstack([m,-m])},),
                method="SLSQP",options={"maxiter":80,"ftol":1e-13})
            controls[2:,j]=result.x
        return (controls[2:]-self.q)/self.h

    def __call__(self,clean,strength):
        z=clean.detach().cpu().numpy().astype(np.float64).reshape(-1,102)
        before,_=self.energy_gradient(z)
        initial=z.copy()
        # Three preconditioned gradient updates per late denoising step.
        for _ in range(3):
            _,gradient=self.energy_gradient(z)
            delta=cho_solve(self.factor,gradient.T,check_finite=False).T
            z-=float(strength)*.35*delta
        after,_=self.energy_gradient(z)
        item={"strength":float(strength),"energy_before":float(before.mean()),
              "energy_after_gradient":float(after.mean()),
              "gradient_update_norm":float(np.linalg.norm(z-initial,axis=1).mean()),
              "prediction":"floating-base local linearization"}
        if strength>=.999:
            free=z.reshape(-1,6,17)*self.scale+self.mean
            if self.diagnostic:
                self.stages["before_final_gradient"]=initial.reshape(-1,6,17)*self.scale+self.mean
                self.stages["after_gradient"]=free.copy()
            corrected=np.array([self.project_joint_bounds(f) for f in free]) if self.project else free
            item["joint_projection_norm_rad_per_s"]=float(np.linalg.norm((corrected-free).reshape(len(z),-1),axis=1).mean())
            z=((corrected-self.mean)/self.scale).reshape(-1,102)
            item["energy_after_projection"]=float(self.energy_gradient(z)[0].mean())
            item["admissible_after_projection"]=sum(SplineChunk.decode(f,self.q,self.dq,self.h).bounds_ok(self.env.spec) for f in corrected)
            if self.diagnostic:
                self.stages["after_projection"]=corrected.copy()
        self.history.append(item)
        return torch.as_tensor(z.reshape(-1,6,17),dtype=clean.dtype,device=clean.device)

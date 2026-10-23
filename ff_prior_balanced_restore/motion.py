"""Exact derivative-control-polygon inequalities; no model propagation."""
import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds
from ff_prior.spline import Reference
from ff_prior.proposal_generator import quick_gates
from .protocol import P


def matrices(q0, duration):
    base=Reference(np.zeros((19,17)),q0,duration)
    offsets=[];maps=[]
    for order in (0,1,2):
        curve=base.curves[order];count=len(curve.t)-curve.k-1
        offsets.append(curve.c[:count]/duration**order)
        columns=[]
        for j in range(19):
            z=np.zeros((19,17));z[j]=1.
            c=Reference(z,q0,duration).curves[order].c[:count]/duration**order
            columns.append((c-offsets[-1])[:,0])
        maps.append(np.asarray(columns).T)
    return maps,offsets


class Motion:
    def __init__(self,robot,q0):
        self.robot=robot;self.q0=np.asarray(q0);self.duration=P['duration_s']
        maps,offsets=matrices(self.q0,self.duration)
        self.maps,self.offsets=maps,offsets
        A=[];lo=[];hi=[]
        for order,limit in ((1,robot.spec.planner_velocity_limits),(2,robot.spec.planner_acceleration_limits)):
            A.append(np.kron(maps[order],np.eye(17)))
            bound=np.broadcast_to(P['motion_fraction']*np.asarray(limit),offsets[order].shape)
            lo.append((-bound-offsets[order]).ravel());hi.append((bound-offsets[order]).ravel())
        limited=robot.model.jnt_limited[robot.joints].astype(bool)
        ranges=robot.model.jnt_range[robot.joints][limited]
        if np.any(limited):
            A.append(np.kron(maps[0],robot.B[limited]))
            offset=offsets[0]@robot.B[limited].T
            lo.append((ranges[:,0]+1e-10-offset).ravel())
            hi.append((ranges[:,1]-1e-10-offset).ravel())
        A=np.vstack(A);lo=np.concatenate(lo);hi=np.concatenate(hi)
        keep=np.any(A!=0.,axis=1)
        assert np.all(lo[~keep]<=0) and np.all(hi[~keep]>=0),'FIXED_INITIAL_MOTION_INFEASIBLE'
        self.A,self.lo,self.hi=A[keep],lo[keep],hi[keep]
        self.lower=np.tile(robot.lower+1e-10,19);self.upper=np.tile(robot.upper-1e-10,19)

    def valid(self,z,tol=1e-9):
        x=np.asarray(z).ravel();y=self.A@x
        return bool(np.isfinite(x).all() and np.all(x>=self.lower-tol) and np.all(x<=self.upper+tol)
                    and np.all(y>=self.lo-tol) and np.all(y<=self.hi+tol)
                    and all(quick_gates(self.robot,Reference(x.reshape(19,17),self.q0,self.duration)).values()))

    def fit(self,times,desired):
        base=Reference(np.zeros((19,17)),self.q0,self.duration).evaluate(times)
        design=[]
        for j in range(19):
            z=np.zeros((19,17));z[j]=1.
            design.append((Reference(z,self.q0,self.duration).evaluate(times)-base)[:,0])
        H=np.asarray(design).T;target=np.asarray(desired)-base
        D=np.diff(np.eye(19),n=2,axis=0)
        def fun(x):
            z=x.reshape(19,17);v=H@z-target
            return float(np.mean(v*v)+P['fit_smooth_weight']*np.mean((D@z)**2))
        def jac(x):
            z=x.reshape(19,17)
            return (2*H.T@(H@z-target)/target.size+
                    2*P['fit_smooth_weight']*D.T@(D@z)/(17*17)).ravel()
        initial=np.tile(self.q0,19);best=initial.copy();best_f=fun(best)
        assert self.valid(initial),'ORIGINAL_CONSTANT_NOT_MOTION_FEASIBLE'
        def callback(x):
            nonlocal best,best_f
            value=fun(x)
            if self.valid(x) and value<best_f:best=x.copy();best_f=value
        sol=minimize(fun,initial,jac=jac,method='SLSQP',bounds=Bounds(self.lower,self.upper),
            constraints=LinearConstraint(self.A,self.lo,self.hi),callback=callback,
            options={'maxiter':P['fit_solver_iterations'],'ftol':1e-12})
        callback(sol.x)
        return best.reshape(19,17),dict(success=bool(sol.success),message=str(sol.message),
            iterations=int(sol.nit),initial_loss=fun(initial),returned_loss=best_f,
            returned_motion_valid=self.valid(best),endpoint_fixed=False,variables=323,
            classification='OPTIMIZATION_INITIAL_GUESS_NOT_MODEL_VALID',new_physics=0)

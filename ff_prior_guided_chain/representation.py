"""Reference-only sparse QP, with exact control-hull motion constraints."""
import numpy as np
import scipy.sparse as sp
from scipy.interpolate import BSpline
import clarabel
from ff_prior.spline import Reference
from ff_prior.storage import digest
from ff_prior_horizon_restore.motion import matrices
from ff_prior_dynamic_margin.representation import polynomial_bounds
from .protocol import P


def solve_qp(H,f,A,lo,hi,deadline):
    import time
    A=sp.csc_matrix(A);lo=np.asarray(lo);hi=np.asarray(hi)
    keep_hi=np.isfinite(hi);keep_lo=np.isfinite(lo)
    C=sp.vstack([A[keep_hi],-A[keep_lo]],format='csc')
    b=np.r_[hi[keep_hi],-lo[keep_lo]]
    settings=clarabel.DefaultSettings();settings.verbose=False
    settings.max_iter=P['solver_iterations'];settings.time_limit=max(.001,deadline-time.time())
    settings.tol_gap_abs=1e-10;settings.tol_gap_rel=1e-10;settings.tol_feas=1e-10
    solver=clarabel.DefaultSolver(sp.triu(sp.csc_matrix(H),format='csc'),np.asarray(f),C,b,
        [clarabel.NonnegativeConeT(len(b))],settings)
    result=solver.solve();x=np.asarray(result.x);y=A@x
    violation=float(max(np.max(lo-y,initial=0),np.max(y-hi,initial=0)))
    good=bool(np.isfinite(x).all() and violation<=1e-9 and str(result.status) in ('Solved','AlmostSolved'))
    report=dict(status=str(result.status),iterations=int(result.iterations),
        primal_violation=violation,feasible=good,objective=float(result.obj_val),
        solver='clarabel_0.11.1',max_iter=P['solver_iterations'])
    if not good:raise ValueError('REFERENCE_QP_UNRESOLVED:'+str(report))
    return x,report


class TemporalReference:
    def __init__(self,control,knots):
        self.control=np.asarray(control);self.knots=np.asarray(knots)
        self.duration=float(knots[-1]);self.hold=0.;self.spline=BSpline(knots,control,3,extrapolate=False)
        self.reference_hash=digest(dict(control=self.control,knots=self.knots))
    def evaluate(self,t,derivative=0):return self.spline(np.clip(t,0,self.duration),nu=derivative)
    def bounds(self):return polynomial_bounds(self.spline)


class Representation:
    def __init__(self,robot,phase,q0,v0=None,a0=None):
        self.robot,self.phase=robot,phase;self.q0=np.asarray(q0)
        self.duration=27. if phase!='C' else 29.
        if phase=='A':
            self.count=19;self.maps,self.offsets=matrices(q0,27.)
        else:
            self.count=270
            # The frozen network has 271 free positions, not a variable-length input.
            self.knots=np.r_[np.zeros(4),np.arange(1,270)*self.duration/270,np.full(4,self.duration)]
            full=BSpline(self.knots,np.eye(273),3)
            derivatives=[full.derivative(k).c[:273-k] for k in (0,1,2)]
            fixed=np.zeros((273,17));fixed[0]=q0
            v0=np.asarray(v0);a0=np.asarray(a0)
            fixed[1]=q0+v0/(derivatives[1][0,1])
            fixed[2]=(a0-derivatives[2][0,:2]@fixed[:2])/derivatives[2][0,2]
            self.fixed=fixed[:3].copy()
            self.maps=[d[:,3:] for d in derivatives]
            self.offsets=[d@fixed for d in derivatives]
        blocks=[];los=[];his=[]
        for order in (0,1,2):
            off=self.offsets[order]
            if order==0:
                low=np.broadcast_to(robot.lower+1e-10,off.shape);high=np.broadcast_to(robot.upper-1e-10,off.shape)
            else:
                limits=np.asarray(robot.spec.planner_velocity_limits if order==1 else robot.spec.planner_acceleration_limits)*P['motion_fraction']
                high=np.broadcast_to(limits,off.shape);low=-high
            blocks.append(sp.kron(sp.csc_matrix(self.maps[order]),sp.eye(17),format='csc'))
            los.append((low-off).ravel());his.append((high-off).ravel())
        limited=robot.model.jnt_limited[robot.joints].astype(bool)
        if limited.any():
            ranges=robot.model.jnt_range[robot.joints][limited];B=robot.B[limited]
            off=self.offsets[0]@B.T
            blocks.append(sp.kron(sp.csc_matrix(self.maps[0]),sp.csc_matrix(B),format='csc'))
            los.append((ranges[:,0]+1e-10-off).ravel());his.append((ranges[:,1]-1e-10-off).ravel())
        self.A=sp.vstack(blocks,format='csc');self.lo=np.concatenate(los);self.hi=np.concatenate(his)
        keep=np.asarray(abs(self.A).sum(axis=1)).ravel()>0
        if np.any(self.lo[~keep]>1e-10) or np.any(self.hi[~keep]<-1e-10):raise ValueError('FIXED_INITIAL_BOUNDARY_INFEASIBLE')
        self.A=self.A[keep];self.lo=self.lo[keep];self.hi=self.hi[keep]
        self.dim=self.count*17
        D=sp.kron(sp.csc_matrix(np.diff(np.eye(self.count),n=2,axis=0)),sp.eye(17),format='csc')
        self.H=sp.eye(self.dim,format='csc')+P['projection_smooth_weight']*(D.T@D)

    def ref(self,x):
        x=np.asarray(x).reshape(self.count,17)
        return Reference(x,self.q0,27.,.5) if self.phase=='A' else TemporalReference(np.vstack([self.fixed,x]),self.knots)
    def free(self,control):return np.asarray(control) if self.phase=='A' else np.asarray(control)[3:]
    def valid(self,x):
        y=self.A@np.asarray(x).ravel()
        return bool(np.isfinite(y).all() and np.all(y>=self.lo-1e-9) and np.all(y<=self.hi+1e-9))
    def project(self,control,deadline):
        target=self.free(control).ravel()
        x,report=solve_qp(self.H,-target,self.A,self.lo,self.hi,deadline)
        report['maximum_change_rad']=float(np.max(abs(x-target)))
        return x.reshape(self.count,17),report

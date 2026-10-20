"""V2: one V7 domain for measured state, ramped reference, and acceptance."""
import numpy as np
from scipy.optimize import minimize
from ff_prior_waypoint_scene.experiment import Environment as OriginalEnvironment


def velocity_bounds(q, reference, previous, lower, upper, vmax, acc):
    """Exact affine reference integration bounds for all ten 2ms ramp steps."""
    lo = np.maximum.reduce([-vmax,4*(lower+.04-q),previous-acc*.02])
    hi = np.minimum.reduce([vmax,4*(upper-.04-q),previous+acc*.02])
    for k in range(1,11):
        weight = .002*k*(k+1)/20
        offset = reference+.002*(k-k*(k+1)/20)*previous
        lo = np.maximum(lo,(lower+.04-offset)/weight)
        hi = np.minimum(hi,(upper-.04-offset)/weight)
    return lo,hi


class Environment(OriginalEnvironment):
    def record(self):
        super().record()  # Save a violating actual frame before fail-closed stop.
        q = self.history["q17"][-1]
        if np.any(q<self.robot.lower) or np.any(q>self.robot.upper):
            raise RuntimeError("ACTUAL_V7_DOMAIN_VIOLATION")
        if np.any(self.reference_q<self.robot.lower) or np.any(self.reference_q>self.robot.upper):
            raise RuntimeError("REFERENCE_V7_DOMAIN_VIOLATION")

    def velocity_filter(self, desired, g, drift, hessian=None):
        q,_ = self.state()
        lo,hi = velocity_bounds(q,self.reference_q,self.command,self.robot.lower,self.robot.upper,
            .70*self.spec.planner_velocity_limits,.60*self.spec.planner_acceleration_limits)
        a,b,margin = self.clearance(g,drift)
        b=b+.008  # unchanged collision planner buffer
        if np.any(lo>hi) or margin < -.001:
            raise RuntimeError("NO_FEASIBLE_V7_DOMAIN_VELOCITY")
        x=np.clip(desired,lo,hi)
        active=len(b) and np.min(a@x-b)<-1e-8
        if active or (hessian is not None and np.max(abs(x-desired))>1e-10):
            h=np.eye(17) if hessian is None else hessian/max(np.max(np.diag(hessian)),1e-8)
            result=minimize(lambda z:.5*(z-desired)@h@(z-desired),x,
                jac=lambda z:h@(z-desired),bounds=list(zip(lo,hi)),method="SLSQP",
                constraints=({"type":"ineq","fun":lambda z:a@z-b,"jac":lambda z:a},) if len(b) else (),
                options={"ftol":1e-12,"maxiter":200})
            x=result.x
            if not result.success or (len(b) and np.min(a@x-b)<-1e-6):
                self.qp_failures+=1
                raise RuntimeError(f"No feasible velocity: {result.message}")
        if not np.isfinite(x).all() or np.any(x<lo-1e-12) or np.any(x>hi+1e-12):
            raise RuntimeError("QP_DOMAIN_BOUND_VIOLATION")
        # Never clip actual qpos or integrated reference to manufacture a pass.
        self.last_filter_delta=float(np.linalg.norm(x-desired))
        return x

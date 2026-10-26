"""Small clamped cubic B-spline; measured position/velocity are exact boundaries."""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import minimize

# The previous 70% control-command cap is not a physical velocity limit.
# A measured boundary can exceed it slightly through servo dynamics. Retain
# 25% planning headroom here; the physical spec and execution cap are unchanged.
VELOCITY_FACTOR = .75


@lru_cache(maxsize=32)
def knots(count):
    if count < 4:
        raise ValueError("At least four cubic control points")
    return np.r_[np.zeros(4), np.linspace(0., 1., count-2)[1:-1], np.ones(4)]


@lru_cache(maxsize=32)
def basis_spline(count):
    return BSpline(knots(count), np.eye(count), 3)


@dataclass
class SplineChunk:
    controls: np.ndarray
    horizon: float

    @classmethod
    def decode(cls, free, q, dq, horizon):
        free = np.asarray(free).reshape(-1, 17)
        k = knots(len(free)+2)
        return cls(np.vstack([q, q+horizon*k[4]*dq/3., q+horizon*free]), horizon)

    def sample(self, time):
        s = np.clip(np.asarray(time)/self.horizon, 0., 1.)
        b = basis_spline(len(self.controls))
        return tuple((b(s, nu=i)@self.controls)/self.horizon**i for i in range(3))

    def derivative_controls(self):
        count = len(self.controls)
        k = knots(count)
        v = 3*np.diff(self.controls,axis=0)/(k[4:count+3]-k[1:count])[:,None]/self.horizon
        a = 2*np.diff(v,axis=0)/(k[4:count+2]-k[2:count])[:,None]/self.horizon
        return v, a

    def bound_report(self, spec):
        q = self.controls
        v, a = self.derivative_controls()
        return {"position": bool(np.all(q >= spec.planner_lower+.04-1e-8) and np.all(q <= spec.planner_upper-.04+1e-8)),
            "velocity": bool(np.all(np.abs(v) <= VELOCITY_FACTOR*spec.planner_velocity_limits+1e-8)),
            "acceleration": bool(np.all(np.abs(a) <= spec.planner_acceleration_limits+1e-8)),
            "max_velocity_ratio": float(np.max(np.abs(v)/(VELOCITY_FACTOR*spec.planner_velocity_limits))),
            "max_acceleration_ratio": float(np.max(np.abs(a)/spec.planner_acceleration_limits))}

    def bounds_ok(self, spec):
        r = self.bound_report(spec)
        return all(r[k] for k in ("position","velocity","acceleration"))


def fit_spline(q_future, initial_dq, horizon, count=8, spec=None):
    q0 = q_future[0]
    k = knots(count)
    fixed = np.stack([q0, q0+horizon*k[4]*initial_dq/3.])
    b = basis_spline(count)(np.linspace(0.,1.,len(q_future)))
    rhs = q_future-b[:,:2]@fixed
    # Fit deviations from constant initial velocity. At a shortened final
    # window there may be fewer samples than controls; minimum-norm absolute
    # coordinates would pull unobserved controls towards world zero.
    greville = np.array([np.mean(k[i+1:i+4]) for i in range(count)])
    baseline = q0 + horizon*greville[:,None]*initial_dq
    controls = baseline.copy()
    controls[2:] += np.linalg.lstsq(b[:,2:],q_future-b@baseline,rcond=None)[0]
    chunk = SplineChunk(controls, horizon)
    if spec is not None and not chunk.bounds_ok(spec):
        # Convex quadratic fit, independent six-dimensional QPs per coordinate.
        identity = SplineChunk(np.eye(count), horizon)
        v, a = identity.derivative_controls()
        matrices = np.vstack([np.eye(count), v, a])
        h = b[:,2:].T@b[:,2:]+1e-10*np.eye(count-2)
        for j in range(17):
            low = np.r_[np.full(count,spec.planner_lower[j]+.04),
                np.full(count-1,-(VELOCITY_FACTOR-.0001)*spec.planner_velocity_limits[j]),
                np.full(count-2,-.9999*spec.planner_acceleration_limits[j])]
            high = np.r_[np.full(count,spec.planner_upper[j]-.04),
                np.full(count-1,(VELOCITY_FACTOR-.0001)*spec.planner_velocity_limits[j]),
                np.full(count-2,.9999*spec.planner_acceleration_limits[j])]
            if np.all(matrices@controls[:,j] >= low-1e-9) and np.all(matrices@controls[:,j] <= high+1e-9):
                continue
            offset = matrices[:,:2]@fixed[:,j]
            f = b[:,2:].T@rhs[:,j]
            z = controls[2:,j].copy()
            m = matrices[:,2:]
            lo, hi = low-offset, high-offset
            solution = minimize(lambda z: .5*z@h@z-f@z, z, jac=lambda z:h@z-f,
                constraints=({"type":"ineq","fun":lambda z:np.r_[m@z-lo,hi-m@z],
                              "jac":lambda z:np.vstack([m,-m])},),
                method="SLSQP",options={"maxiter":100,"ftol":1e-12})
            controls[2:,j] = solution.x
    error = float(np.max(np.abs(b@controls-q_future)))
    return ((controls[2:]-q0)/horizon).astype(np.float32), error

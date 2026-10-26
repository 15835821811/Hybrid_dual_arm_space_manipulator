"""A cubic Bezier is a clamped, single-span cubic B-spline.

One second, 17 coordinates, four control points; the first two are determined
by measured position and velocity. Diffusion generates the remaining 2 x 17.
"""
from dataclasses import dataclass
import numpy as np

HORIZON = 1.0
REPLAN = 0.10


def basis(s):
    s = np.asarray(s)
    return np.stack([(1-s)**3, 3*s*(1-s)**2, 3*s*s*(1-s), s**3], axis=-1)


@dataclass
class JointChunk:
    controls: np.ndarray
    horizon: float = HORIZON

    @classmethod
    def decode(cls, free, q, dq, horizon=HORIZON):
        free = np.asarray(free).reshape(2, 17)
        return cls(np.vstack([q, q+horizon*dq/3., q+horizon*free]), horizon)

    def sample(self, t):
        s = np.clip(np.asarray(t) / self.horizon, 0., 1.)
        q = basis(s) @ self.controls
        d = 3*np.diff(self.controls, axis=0) / self.horizon
        db = np.stack([(1-s)**2, 2*s*(1-s), s*s], axis=-1)
        dq = db @ d
        a = 2*np.diff(d, axis=0) / self.horizon
        ddq = np.stack([1-s, s], axis=-1) @ a
        return q, dq, ddq

    def bounds_ok(self, spec):
        """Convex-hull bounds hold throughout this polynomial, not just at knots."""
        q = self.controls
        v = 3*np.diff(q, axis=0)/self.horizon
        a = 2*np.diff(v, axis=0)/self.horizon
        return bool(np.all(q >= spec.planner_lower+.04) and
                    np.all(q <= spec.planner_upper-.04) and
                    np.all(np.abs(v) <= .70*spec.planner_velocity_limits+1e-8) and
                    np.all(np.abs(a) <= spec.planner_acceleration_limits+1e-8))


def fit_chunk(q_future, initial_dq, dt=.02):
    horizon = (len(q_future)-1)*dt
    q0 = q_future[0]
    fixed = np.stack([q0, q0 + horizon*initial_dq/3.])
    b = basis(np.linspace(0., 1., len(q_future)))
    free = np.linalg.lstsq(b[:, 2:], q_future - b[:, :2] @ fixed, rcond=None)[0]
    controls = np.vstack([fixed, free])
    error = float(np.max(np.abs(b @ controls - q_future)))
    return ((free-q0)/horizon).astype(np.float32), error

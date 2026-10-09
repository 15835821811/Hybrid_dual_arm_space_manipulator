"""32x17 clamped cubic B splines with algebraically fixed start boundaries."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline

from v6_4.contracts import CONTROL_POINT_COUNT, DURATION_S, PLANNER_DIM, REPRESENTATION_VERSION


class CubicBSplineCodec:
    """Only C2..C31 are generated; C0 and C1 are eliminated, never clipped."""

    def __init__(self, q0: np.ndarray, dq0: np.ndarray, *, duration_s: float = DURATION_S,
                 n_control_points: int = CONTROL_POINT_COUNT) -> None:
        if duration_s != DURATION_S or n_control_points != CONTROL_POINT_COUNT:
            raise ValueError("V6.4-A fixes 27 seconds and 32 cubic control points")
        self.q0, self.dq0 = self._vector(q0, "q0"), self._vector(dq0, "dq0")
        self.q0.setflags(write=False); self.dq0.setflags(write=False)
        self.duration_s, self.n_control_points, self.degree = float(duration_s), int(n_control_points), 3
        # 32 coefficients, degree three => 29 nonzero uniform knot spans.
        interior = np.linspace(0., self.duration_s, self.n_control_points-self.degree+1)[1:-1]
        self.knots = np.r_[np.zeros(4), interior, np.full(4, self.duration_s)]
        self.knots.setflags(write=False)
        self.fixed_controls = np.stack([self.q0, self.q0 + (self.knots[4]/3.)*self.dq0])
        self.fixed_controls.setflags(write=False)
        self.free_shape = (self.n_control_points-2, PLANNER_DIM)

    @staticmethod
    def _vector(value: np.ndarray, name: str) -> np.ndarray:
        result = np.asarray(value, dtype=float)
        if result.shape != (PLANNER_DIM,) or not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must be a finite 17-vector")
        return result.copy()

    def _controls(self, value: np.ndarray) -> np.ndarray:
        result = np.asarray(value, dtype=float)
        if result.shape != (self.n_control_points, PLANNER_DIM) or not np.all(np.isfinite(result)):
            raise ValueError("full spline coefficients must be finite 32x17 values")
        if not np.array_equal(result[:2], self.fixed_controls):
            raise ValueError("full coefficients do not satisfy the eliminated start boundary exactly")
        return result

    def decode_free(self, controls_free: np.ndarray) -> np.ndarray:
        values = np.asarray(controls_free, dtype=float)
        if values.shape != self.free_shape or not np.all(np.isfinite(values)):
            raise ValueError("free spline coefficients must be finite 30x17 values")
        return np.vstack([self.fixed_controls, values])

    def encode_free(self, control_points: np.ndarray) -> np.ndarray:
        return self._controls(control_points)[2:].copy()

    def basis(self, times: np.ndarray, derivative: int = 0) -> np.ndarray:
        values = np.asarray(times, dtype=float)
        if (values.ndim != 1 or not np.all(np.isfinite(values)) or np.any(values < 0.)
                or np.any(values > self.duration_s) or derivative not in (0, 1, 2, 3)):
            raise ValueError("spline times must be finite within [0,27], derivative 0..3")
        return BSpline(self.knots, np.eye(self.n_control_points), self.degree, extrapolate=False)(values, nu=derivative)

    def sample(self, control_points: np.ndarray, times: np.ndarray) -> dict[str, np.ndarray]:
        values = self._controls(control_points)
        return {name: self.basis(times, derivative) @ values
                for name, derivative in (("q",0), ("dq",1), ("ddq",2))}

    def fit(self, times: np.ndarray, q: np.ndarray, *, regularization: float = 1e-10,
            derivative_samples: np.ndarray | None = None,
            derivative_weight: float = 0.) -> np.ndarray:
        """Constrained least squares; callers record its planner/repair role."""
        times = np.asarray(times, dtype=float); values = np.asarray(q, dtype=float)
        if (values.shape != (len(times), PLANNER_DIM) or len(times) < self.n_control_points
                or not np.all(np.isfinite(values)) or np.any(np.diff(times) <= 0)
                or not np.isfinite(regularization) or regularization < 0.
                or not np.isfinite(derivative_weight) or derivative_weight < 0.):
            raise ValueError("finite ordered fitting samples and nonnegative penalties required")
        basis = self.basis(times)
        matrix, target = basis[:,2:], values - basis[:,:2] @ self.fixed_controls
        if derivative_samples is not None and derivative_weight:
            rates = np.asarray(derivative_samples, dtype=float)
            if rates.shape != values.shape or not np.all(np.isfinite(rates)):
                raise ValueError("fitting derivative samples have invalid shape or values")
            derivative = self.basis(times, 1)
            matrix = np.vstack([matrix, np.sqrt(derivative_weight)*derivative[:,2:]])
            target = np.vstack([target, np.sqrt(derivative_weight)*(rates-derivative[:,:2] @ self.fixed_controls)])
        if regularization:
            # A curvature prior on all coefficients, including the fixed part.
            differences = np.diff(np.eye(self.n_control_points), n=2, axis=0)
            matrix = np.vstack([matrix, np.sqrt(regularization)*differences[:,2:]])
            target = np.vstack([target, -np.sqrt(regularization)*differences[:,:2] @ self.fixed_controls])
        free, _, rank, _ = np.linalg.lstsq(matrix, target, rcond=None)
        if rank != self.n_control_points-2:
            raise ValueError("sample grid does not identify all free spline coefficients")
        return self.decode_free(free)

    def to_dict(self) -> dict:
        return {"representation": REPRESENTATION_VERSION, "degree": self.degree,
                "duration_s": self.duration_s, "control_point_shape": [32,17], "free_shape": [30,17],
                "q0": self.q0.tolist(), "dq0": self.dq0.tolist(), "knots_s": self.knots.tolist(),
                "start_boundary": "C0=q0; C1=q0+(first_nonzero_knot/3)*dq0",
                "clipping_or_projection": False}

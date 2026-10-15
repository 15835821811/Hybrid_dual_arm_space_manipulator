"""One analytic decoder for generation, learning, propagation and execution."""
from dataclasses import dataclass
import numpy as np
from scipy.interpolate import BSpline
from .storage import digest

KNOTS = np.r_[np.zeros(4), np.arange(1, 21) / 21, np.ones(4)]


def decode(z, q0):
    z, q0 = np.asarray(z, float), np.asarray(q0, float)
    if z.shape != (19, 17) or q0.shape != (17,) or not np.isfinite(z).all() or not np.isfinite(q0).all():
        raise ValueError("finite Z[19,17] and q0[17] required")
    return np.vstack([np.repeat(q0[None], 3, axis=0), z[:18], np.repeat(z[-1:], 3, axis=0)])


@dataclass
class Reference:
    z: np.ndarray
    q0: np.ndarray
    duration: float
    hold: float = 0.5

    def __post_init__(self):
        if self.duration <= 0 or self.hold < 0 or self.duration + self.hold > 30:
            raise ValueError("motion+hold must fit 30s deadline")
        self.z, self.q0 = np.asarray(self.z, float).copy(), np.asarray(self.q0, float).copy()
        self.control = decode(self.z, self.q0)
        self.z.setflags(write=False)
        self.q0.setflags(write=False)
        self.control.setflags(write=False)
        curve = BSpline(KNOTS, self.control, 3, extrapolate=False)
        self.curves = (curve, curve.derivative(1), curve.derivative(2))

    @property
    def reference_hash(self):
        return digest({"C": self.control, "knots": KNOTS, "degree": 3, "T": self.duration, "hold": self.hold})

    def evaluate(self, time, derivative=0):
        t = np.asarray(time)
        result = self.curves[derivative](np.clip(t / self.duration, 0, 1)) / self.duration**derivative
        if derivative:
            result = np.where(((t < 0) | (t > self.duration))[..., None], 0, result)
        return result

    def bounds(self, derivative=0):
        curve = self.curves[derivative]
        active = curve.c[:len(curve.t) - curve.k - 1] / self.duration**derivative
        return active.min(axis=0), active.max(axis=0)

    def arrays(self):
        times = np.linspace(0, self.duration, 128)
        return {"Z": self.z, "C": self.control, "knots": KNOTS, "degree": np.array(3),
                "T_ref": np.array(self.duration), "T_hold": np.array(self.hold),
                "view_time": times, "view_q": self.evaluate(times),
                "view_qd": self.evaluate(times, 1), "view_qdd": self.evaluate(times, 2)}


def from_arrays(arrays):
    ref = Reference(arrays["Z"], arrays["C"][0], float(arrays["T_ref"]), float(arrays["T_hold"]))
    if not np.array_equal(ref.control, arrays["C"]) or not np.array_equal(KNOTS, arrays["knots"]):
        raise ValueError("saved spline decoder mismatch")
    return ref

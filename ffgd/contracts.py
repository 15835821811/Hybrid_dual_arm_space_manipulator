"""Immutable, content-addressed numerical inputs. No solution-path fields."""
from dataclasses import dataclass
import hashlib
import numpy as np


def immutable(value):
    a = np.ascontiguousarray(value, dtype=np.float64)
    if not np.isfinite(a).all():
        raise ValueError("nonfinite input")
    # Immutable bytes backing prevents callers re-enabling WRITEABLE.
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def array_hash(a):
    return hashlib.sha256(str(a.shape).encode() + a.tobytes()).hexdigest()


@dataclass(frozen=True)
class Candidate:
    control: np.ndarray
    duration: float
    seed: int

    def __post_init__(self):
        object.__setattr__(self, "control", immutable(self.control))
        if self.control.ndim != 2 or not np.isfinite(self.duration) or self.duration <= 0:
            raise ValueError("invalid trajectory")

    @property
    def identity(self):
        return hashlib.sha256((array_hash(self.control) + repr(self.duration) + str(self.seed)).encode()).hexdigest()


@dataclass(frozen=True)
class Snapshot:
    model_hash: str
    mask: int
    state: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "state", immutable(self.state))

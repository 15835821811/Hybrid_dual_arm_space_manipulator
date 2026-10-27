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


@dataclass(frozen=True)
class PoseConstraint:
    time: float
    tip: int
    position: np.ndarray
    rotation: np.ndarray
    position_tolerance: float = .005
    rotation_tolerance: float = np.pi/180
    position_enabled: bool = True
    rotation_enabled: bool = True

    def __post_init__(self):
        object.__setattr__(self,'position',immutable(self.position))
        object.__setattr__(self,'rotation',immutable(self.rotation))
        if self.position.shape!=(3,) or self.rotation.shape!=(3,3) or self.tip not in (0,1):raise ValueError('invalid pose')
        if not np.isfinite([self.position_tolerance,self.rotation_tolerance,self.time]).all() or self.position_tolerance<=0 or self.rotation_tolerance<=0 or self.time<0:raise ValueError('invalid tolerance/time')
        if not np.allclose(self.rotation.T@self.rotation,np.eye(3),atol=1e-10) or np.linalg.det(self.rotation)<0:raise ValueError('not SO(3)')


@dataclass(frozen=True)
class TaskSpec:
    duration: float
    poses: tuple

    def __post_init__(self):
        object.__setattr__(self,'poses',tuple(self.poses))
        if self.duration<=0 or not np.isfinite(self.duration) or not self.poses:raise ValueError('invalid task')
        if any(p.time>self.duration for p in self.poses):raise ValueError('task beyond duration')


@dataclass(frozen=True)
class BoxObstacle:
    center: np.ndarray
    half_size: np.ndarray
    rotation: np.ndarray

    def __post_init__(self):
        for field in ('center','half_size','rotation'):object.__setattr__(self,field,immutable(getattr(self,field)))
        if self.center.shape!=(3,) or self.half_size.shape!=(3,) or np.any(self.half_size<=0):raise ValueError('invalid obstacle dimensions')
        if self.rotation.shape!=(3,3) or not np.allclose(self.rotation.T@self.rotation,np.eye(3),atol=1e-10) or np.linalg.det(self.rotation)<0:raise ValueError('invalid obstacle rotation')

    @property
    def identity(self):return array_hash(np.r_[self.center,self.half_size,self.rotation.ravel()])


@dataclass(frozen=True)
class SceneSpec:
    boxes: tuple = ()

    def __post_init__(self):
        object.__setattr__(self,'boxes',tuple(sorted(self.boxes,key=lambda box:box.identity)))

    @property
    def identity(self):return hashlib.sha256(''.join(b.identity for b in self.boxes).encode()).hexdigest()

"""Physical-time cubic B-splines and deterministic affine boundary codec."""
from dataclasses import dataclass
import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import qr


def knots(n, degree=3):
    if n < degree + 1:
        raise ValueError("not enough control points")
    return np.r_[np.zeros(degree + 1), np.arange(1, n-degree)/(n-degree), np.ones(degree+1)]


def basis(n, times, duration, derivative=0):
    t = np.asarray(times, dtype=float)
    if not np.isfinite(duration) or duration <= 0 or not np.isfinite(t).all():
        raise ValueError("invalid physical time")
    if np.any(t < 0) or np.any(t > duration) or derivative not in (0, 1, 2, 3):
        raise ValueError("time or derivative outside domain")
    return BSpline(knots(n), np.eye(n), 3)(t/duration, nu=derivative) / duration**derivative


@dataclass(frozen=True)
class Boundary:
    time: float
    derivative: int
    joints: tuple
    values: tuple


class BoundaryCodec:
    """Flattened control-major constraints; full nullspace, no learned truncation.

    Pivoted elimination is deterministic for the fixed boundary template.
    Dependent consistent rows are accepted; conflicting rows are rejected.
    """
    def __init__(self, n, dof, duration, boundaries, origin='pivot_zero'):
        if origin not in ('pivot_zero','initial_position'):raise ValueError('unknown boundary coordinate origin')
        boundaries=tuple(boundaries)
        self.origin=origin
        self.n, self.dof, self.duration = n, dof, float(duration)
        rows, rhs = [], []
        for bc in boundaries:
            if len(bc.joints) != len(bc.values):
                raise ValueError("boundary mask/value mismatch")
            b = basis(n, bc.time, duration, bc.derivative)
            for j, value in zip(bc.joints, bc.values):
                if j < 0 or j >= dof:
                    raise ValueError("invalid joint mask")
                row = np.zeros((n, dof)); row[:, j] = b
                rows.append(row.ravel()); rhs.append(value)
        self.A = np.array(rows).reshape(-1, n*dof)
        self.b = np.array(rhs, dtype=float)
        if not np.isfinite(self.b).all():
            raise ValueError("nonfinite boundary")
        if len(rows):
            # Normalize physical derivative rows before rank decisions. Select
            # earliest independent columns per joint, avoiding QR pivot ties
            # that could reorder labels when duration rescales derivative rows.
            scales=np.linalg.norm(self.A,axis=1)
            if np.any(scales==0):raise ValueError('zero boundary row')
            normalized=self.A/scales[:,None];normalized_b=self.b/scales
            selected=[]
            for joint in range(dof):
                local=normalized[:,joint::dof];local=local[np.any(local!=0,axis=1)]
                columns=[];rank=0
                for column in range(n):
                    newrank=np.linalg.matrix_rank(local[:,columns+[column]],tol=1e-12) if len(local) else 0
                    if newrank>rank:columns.append(column);rank=newrank
                selected.extend(column*dof+joint for column in columns)
            dep=np.asarray(sorted(selected),dtype=int)
            free=np.setdiff1d(np.arange(n*dof),dep);self.rank=len(dep)
            AD = normalized[:, dep]
            particular = np.linalg.lstsq(AD, normalized_b, rcond=None)[0]
            self.offset = np.zeros(n*dof); self.offset[dep] = particular
            N = np.zeros((n*dof, len(free))); N[free, np.arange(len(free))] = 1
            N[dep] = -np.linalg.lstsq(AD, normalized[:, free], rcond=None)[0]
            # Deterministic orthonormal basis, canonical signs.
            self.N, _ = np.linalg.qr(N, mode="reduced")
            for k, i in enumerate(free):
                if self.N[i, k] < 0: self.N[:, k] *= -1
            if np.max(np.abs(self.A @ self.offset - self.b)) > 1e-10:
                raise ValueError("inconsistent boundary constraints")
        else:
            self.rank = 0; self.offset = np.zeros(n*dof); self.N = np.eye(n*dof)
        self.free_dim = self.N.shape[1]
        if origin=='initial_position':
            # Change only the affine coordinate origin, never the admissible
            # trajectory set or its full nullspace. Anchor uses initial position
            # only; it receives no goal, historical path or scene identifier.
            initial={j:value for bc in boundaries if bc.time==0 and bc.derivative==0 for j,value in zip(bc.joints,bc.values)}
            if set(initial)!=set(range(dof)):raise ValueError('initial-position origin requires complete initial position')
            anchor=np.tile([initial[j] for j in range(dof)],n)
            self.offset=self.offset+self.N@(self.N.T@(anchor-self.offset))
            if np.max(abs(self.A@self.offset-self.b))>1e-10:raise ValueError('coordinate origin violates boundary')

    def decode(self, z):
        if z.shape[-1] != self.free_dim:
            raise ValueError("free dimension mismatch")
        if type(z).__module__.startswith("torch"):
            import torch
            C = torch.as_tensor(self.offset, dtype=z.dtype, device=z.device) + z @ torch.as_tensor(self.N.T, dtype=z.dtype, device=z.device)
        else:
            C = self.offset + np.asarray(z) @ self.N.T
        return C.reshape(*z.shape[:-1], self.n, self.dof)

    def encode(self, control):
        flat = np.asarray(control).reshape(-1, self.n*self.dof)
        if self.A.size and np.max(abs(flat @ self.A.T - self.b)) > 1e-10:
            raise ValueError("control violates boundary")
        return ((flat-self.offset) @ self.N).reshape(*np.asarray(control).shape[:-2], self.free_dim)

    def evaluate(self, z, times, derivative=0):
        C = self.decode(z)
        B = basis(self.n, times, self.duration, derivative)
        if type(C).__module__.startswith("torch"):
            import torch
            return torch.as_tensor(B, dtype=C.dtype, device=C.device) @ C
        return B @ C


class AffineScaler:
    def fit(self, values, split):
        if split != "TRAIN": raise ValueError("scaler may only fit TRAIN")
        self.mean = np.mean(values, axis=0)
        self.scale = np.maximum(np.std(values, axis=0), 1e-8)
        return self

    def encode(self, values): return (values-self.mean)/self.scale
    def decode(self, values): return values*self.scale+self.mean

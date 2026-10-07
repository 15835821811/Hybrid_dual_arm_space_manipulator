"""Private positions-only PCC evaluation at one immutable arc-length grid.

The state-dependent section transforms and points use the same arithmetic as
the existing batched point model. Only constant arc indexing is cached, and
rotation/Jacobian result objects unused by the envelope are not constructed.
"""

from __future__ import annotations

import numpy as np

from v6_lite.continuum_model_spec import CONTINUUM_PLANNER_DOF
from v6_lite.continuum_shape_model import ContinuumShapeModel, skew, validate_transform


class FixedArcPCCPositions:
    def __init__(self, spec, arclengths: np.ndarray) -> None:
        self.spec = spec
        self.shape = ContinuumShapeModel(spec)
        requested = np.asarray(arclengths, dtype=np.float64)
        boundaries = spec.segment_boundaries_m
        total = float(boundaries[-1])
        if (requested.ndim != 1 or np.any(~np.isfinite(requested))
                or np.any(requested < -1e-12)
                or np.any(requested > total + 1e-12)):
            raise ValueError("fixed arclengths must lie in the PCC shape")
        clipped = np.clip(requested, 0.0, total)
        segment_ids = np.minimum(
            np.searchsorted(boundaries[1:], clipped, side="right"),
            len(spec.segment_lengths_m) - 1,
        )
        local_s = clipped - boundaries[segment_ids]
        local_s[clipped >= total] = spec.segment_lengths_m[-1]
        self._sections = tuple(
            (segment_id, np.flatnonzero(segment_ids == segment_id),
             float(length))
            for segment_id, length in enumerate(spec.segment_lengths_m)
        )
        self._local_s = local_s
        self._count = len(requested)

    def evaluate(self, planner_configuration: np.ndarray,
                 base_transform: np.ndarray) -> np.ndarray:
        q = np.asarray(planner_configuration, dtype=np.float64)
        if q.shape != (CONTINUUM_PLANNER_DOF,) or np.any(~np.isfinite(q)):
            raise ValueError("PCC configuration must be finite with shape (10,)")
        base = validate_transform(base_transform)
        prefixes = [base @ self.spec.base_to_shape_start]
        for segment_id, length in enumerate(self.spec.segment_lengths_m):
            full = self.shape._section_transform(
                q[2 * segment_id:2 * segment_id + 2], float(length),
                float(length), self.spec.pcc_bending_map,
            )
            prefixes.append(prefixes[-1] @ full)
        positions = np.empty((self._count, 3), dtype=np.float64)
        tangent = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        for segment_id, indexes, length in self._sections:
            if not len(indexes):
                continue
            u = self._local_s[indexes]
            omega = (self.spec.pcc_bending_map
                     @ q[2 * segment_id:2 * segment_id + 2]) / length
            generator = skew(omega)
            generator2 = generator @ generator
            magnitude = float(np.linalg.norm(omega))
            if magnitude * length <= 1e-4:
                magnitude2 = magnitude * magnitude
                b = (u**2 / 2.0 - magnitude2 * u**4 / 24.0
                     + magnitude2**2 * u**6 / 720.0)
                c = (u**3 / 6.0 - magnitude2 * u**5 / 120.0
                     + magnitude2**2 * u**7 / 5040.0)
            else:
                angle = magnitude * u
                sine = np.sin(angle)
                b = (1.0 - np.cos(angle)) / (magnitude * magnitude)
                c = (angle - sine) / (magnitude**3)
            local_positions = (
                u[:, None] * tangent
                + b[:, None] * (generator @ tangent)
                + c[:, None] * (generator2 @ tangent)
            )
            prefix = prefixes[segment_id]
            positions[indexes] = (local_positions @ prefix[:3, :3].T
                                  + prefix[:3, 3])
        return positions

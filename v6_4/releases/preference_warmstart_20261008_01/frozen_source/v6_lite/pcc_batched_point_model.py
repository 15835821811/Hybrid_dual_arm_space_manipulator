"""Read-only vectorized PCC point batch for bounded-query performance trials.

The existing shape model remains the reference. This class changes only the
point/rotation evaluation path when Jacobians are not requested; the analytic
Jacobian path delegates to the reference implementation. No controller selects
this model. An offline parity and timing audit is required before considering
it for a shadow or online query.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from v6_lite.continuum_model_spec import CONTINUUM_PLANNER_DOF
from v6_lite.continuum_shape_model import (
    ContinuumPoint, ContinuumShapeModel, skew, validate_transform,
)


class BatchedPointContinuumShapeModel(ContinuumShapeModel):
    """Calculate all no-Jacobian section points from shared full prefixes."""

    def batch_query(
        self,
        planner_configuration: np.ndarray,
        base_transform: np.ndarray,
        arclengths: Iterable[float],
        *,
        with_jacobians: bool = False,
        with_rotation_jacobians: bool = True,
    ) -> tuple[ContinuumPoint, ...]:
        if with_jacobians:
            return super().batch_query(
                planner_configuration, base_transform, arclengths,
                with_jacobians=True,
                with_rotation_jacobians=with_rotation_jacobians,
            )
        q = np.asarray(planner_configuration, dtype=np.float64)
        if q.shape != (CONTINUUM_PLANNER_DOF,) or np.any(~np.isfinite(q)):
            raise ValueError("PCC configuration must be finite with shape (10,)")
        base = validate_transform(base_transform)
        requested = np.asarray([float(value) for value in arclengths],
                               dtype=np.float64)
        boundaries = self.spec.segment_boundaries_m
        total = float(boundaries[-1])
        if (np.any(~np.isfinite(requested)) or np.any(requested < -1e-12)
                or np.any(requested > total + 1e-12)):
            raise ValueError(f"arclength must lie in [0, {total}]")
        clipped = np.clip(requested, 0.0, total)
        segment_ids = np.minimum(
            np.searchsorted(boundaries[1:], clipped, side="right"),
            len(self.spec.segment_lengths_m) - 1,
        )
        local_s = clipped - boundaries[segment_ids]
        local_s[clipped >= total] = self.spec.segment_lengths_m[-1]

        prefixes = [base @ self.spec.base_to_shape_start]
        for segment_id, length in enumerate(self.spec.segment_lengths_m):
            full = self._section_transform(
                q[2 * segment_id:2 * segment_id + 2], float(length),
                float(length), self.spec.pcc_bending_map,
            )
            prefixes.append(prefixes[-1] @ full)

        positions = np.empty((len(requested), 3), dtype=np.float64)
        rotations = np.empty((len(requested), 3, 3), dtype=np.float64)
        tangent = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        identity = np.eye(3, dtype=np.float64)
        for segment_id, length in enumerate(self.spec.segment_lengths_m):
            indexes = np.flatnonzero(segment_ids == segment_id)
            if not len(indexes):
                continue
            u = local_s[indexes]
            omega = (self.spec.pcc_bending_map
                     @ q[2 * segment_id:2 * segment_id + 2]) / float(length)
            generator = skew(omega)
            generator2 = generator @ generator
            magnitude = float(np.linalg.norm(omega))
            if magnitude * float(length) <= 1e-4:
                magnitude2 = magnitude * magnitude
                a = u - magnitude2 * u**3 / 6.0 + magnitude2**2 * u**5 / 120.0
                b = u**2 / 2.0 - magnitude2 * u**4 / 24.0 + magnitude2**2 * u**6 / 720.0
                c = u**3 / 6.0 - magnitude2 * u**5 / 120.0 + magnitude2**2 * u**7 / 5040.0
            else:
                angle = magnitude * u
                sine = np.sin(angle)
                a = sine / magnitude
                b = (1.0 - np.cos(angle)) / (magnitude * magnitude)
                c = (angle - sine) / (magnitude**3)
            local_positions = (
                u[:, None] * tangent
                + b[:, None] * (generator @ tangent)
                + c[:, None] * (generator2 @ tangent)
            )
            local_rotations = (identity[None]
                               + a[:, None, None] * generator
                               + b[:, None, None] * generator2)
            prefix = prefixes[segment_id]
            positions[indexes] = (local_positions @ prefix[:3, :3].T
                                  + prefix[:3, 3])
            rotations[indexes] = np.einsum(
                "ab,nbc->nac", prefix[:3, :3], local_rotations,
            )
        return tuple(
            ContinuumPoint(
                arclength_m=float(s), segment_index=int(segment_id),
                segment_arclength_m=float(section_s),
                position=positions[i].copy(), rotation=rotations[i].copy(),
                position_jacobian=np.zeros((3, 10), dtype=np.float64),
                rotation_jacobian=np.zeros((3, 10), dtype=np.float64),
            )
            for i, (s, segment_id, section_s) in enumerate(
                zip(requested, segment_ids, local_s)
            )
        )

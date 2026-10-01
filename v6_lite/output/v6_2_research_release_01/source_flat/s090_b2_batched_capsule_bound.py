"""Private batched capsule midpoint bounds with the original exact winner query.

Only the lower-bound screening calculation is batched. Every capsule remains
eligible and an exact segment/OBB query decides the minimum. A 1 pm downward
rounding guard affects screening only, not the reported clearance or margins.
"""

from __future__ import annotations

import numpy as np

from v6_lite.shape_clearance import (
    CapsuleEnvelopeSet, ClearanceResult, OrientedBox,
    capsule_clearance_to_obb,
)


class BatchedCapsuleMidpointBounds:
    def __init__(self, envelopes: CapsuleEnvelopeSet) -> None:
        if not envelopes.capsules:
            raise ValueError("capsule envelope set is empty")
        self.envelopes = envelopes
        capsules = envelopes.capsules
        self._body_ids = np.asarray([item.body_id for item in capsules],
                                    dtype=np.int32)
        self._local_midpoints = np.asarray([
            0.5 * (item.local_start + item.local_end) for item in capsules],
            dtype=np.float64)
        self._half_lengths = np.asarray([
            0.5 * item.axis_length_m for item in capsules], dtype=np.float64)
        self._radii = np.asarray([item.radius_m for item in capsules],
                                 dtype=np.float64)

    def minimum(self, model, data, box: OrientedBox) -> ClearanceResult:
        rotations = np.asarray(data.xmat[self._body_ids],
                               dtype=np.float64).reshape(-1, 3, 3)
        origins = np.asarray(data.xpos[self._body_ids], dtype=np.float64)
        midpoints = origins + np.einsum(
            "nij,nj->ni", rotations, self._local_midpoints)
        local = (midpoints - box.center) @ box.rotation
        closest = np.clip(local, -box.half_extents, box.half_extents)
        outside = np.linalg.norm(local - closest, axis=1)
        interior = -np.min(box.half_extents - np.abs(local), axis=1)
        signed = np.where(outside > 1e-14, outside, interior)
        lower_bounds = signed - self._half_lengths - self._radii - 1e-12
        order = np.argsort(lower_bounds)
        capsules = self.envelopes.capsules
        minimum = capsule_clearance_to_obb(
            model, data, capsules[int(order[0])], box)
        for raw_index in order[1:]:
            index = int(raw_index)
            if lower_bounds[index] > minimum.signed_distance_m + 1e-12:
                break
            candidate = capsule_clearance_to_obb(
                model, data, capsules[index], box)
            if candidate.signed_distance_m < minimum.signed_distance_m:
                minimum = candidate
        return minimum

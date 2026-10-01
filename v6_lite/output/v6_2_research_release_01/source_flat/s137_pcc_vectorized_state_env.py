"""Private batched evaluation of the unchanged state-local PCC tube check.

The capsule fit, five axis samples, 17 samples per PCC section, radii,
sampling allowance and arithmetic pad are inherited from the reference
evaluator. Only the independent capsule distance arithmetic is batched.
This class is not selected by the production controller.
"""

from __future__ import annotations

import numpy as np

from v6_lite.pcc_clearance import V61A_PCC_TUBE_RADII_M
from v6_lite.pcc_prepared_state_envelope import (
    PreparedStateLocalPCCEnvelopeAudit,
)
from v6_lite.pcc_state_local_envelope import (
    ARITHMETIC_PAD_M, CAPSULE_AXIS_SAMPLES, PCC_SAMPLES_PER_SEGMENT,
    CapsuleContainment, StateLocalEnvelopeResult,
)


class VectorizedStateLocalPCCEnvelopeAudit(
        PreparedStateLocalPCCEnvelopeAudit):
    """Batch only per-capsule sampled-axis distances at each actual state."""

    def __init__(self, model, spec) -> None:
        super().__init__(model, spec)
        capsules = self.envelopes.capsules
        parameters = np.linspace(0.0, 1.0, CAPSULE_AXIS_SAMPLES)
        self._local_axis = np.asarray([
            ((1.0 - parameters[:, None]) * item.local_start
             + parameters[:, None] * item.local_end)
            for item in capsules
        ], dtype=np.float64)
        self._body_ids = np.asarray([item.body_id for item in capsules],
                                    dtype=np.int32)
        self._segment_ids = np.asarray([item.segment_index for item in capsules],
                                       dtype=np.int32)
        self._allowance_m = np.asarray([
            item.axis_length_m / (2.0 * (CAPSULE_AXIS_SAMPLES - 1))
            for item in capsules
        ], dtype=np.float64)
        self._radius_m = np.asarray([item.radius_m for item in capsules],
                                    dtype=np.float64)
        self._excess_m = np.asarray([
            max(0.0, item.maximum_vertex_excess_m) for item in capsules
        ], dtype=np.float64)
        self._declared_m = np.asarray([
            V61A_PCC_TUBE_RADII_M[item.segment_index] for item in capsules
        ], dtype=np.float64)

    def evaluate(self, data, planner_q, base_transform) -> StateLocalEnvelopeResult:
        points = self.shape.batch_query(
            planner_q, base_transform, self.arclengths, with_jacobians=False,
        )
        pcc = np.asarray([item.position for item in points],
                         dtype=np.float64).reshape(
            len(self.spec.segment_lengths_m), PCC_SAMPLES_PER_SEGMENT, 3,
        )
        rotation = np.asarray(data.xmat[self._body_ids],
                              dtype=np.float64).reshape(-1, 3, 3)
        origin = np.asarray(data.xpos[self._body_ids], dtype=np.float64)
        axis = origin[:, None, :] + np.matmul(
            self._local_axis, np.transpose(rotation, (0, 2, 1)),
        )
        distances = np.linalg.norm(
            axis[:, :, None, :] - pcc[self._segment_ids][:, None, :, :],
            axis=3,
        )
        sampled_max = np.max(np.min(distances, axis=2), axis=1)
        required = np.nextafter(
            sampled_max + self._allowance_m + self._radius_m
            + self._excess_m + ARITHMETIC_PAD_M, np.inf,
        )
        margins = self._declared_m - required
        rows = tuple(CapsuleContainment(
            geom_name=item.geom_name,
            segment_id=item.segment_index,
            sampled_axis_to_pcc_max_m=float(sampled_max[index]),
            axis_sampling_allowance_m=float(self._allowance_m[index]),
            fitted_capsule_radius_m=item.radius_m,
            required_tube_radius_upper_m=float(required[index]),
            declared_tube_radius_m=float(self._declared_m[index]),
            margin_m=float(margins[index]),
        ) for index, item in enumerate(self.envelopes.capsules))
        minimum = float(np.min(margins))
        return StateLocalEnvelopeResult(
            status=("COVERED_AT_THIS_STATE" if minimum >= 0.0
                    else "NOT_COVERED_AT_THIS_STATE"),
            min_margin_m=minimum,
            capsules=rows,
            fallback_geom_names=self.envelopes.fallback_geom_names,
        )

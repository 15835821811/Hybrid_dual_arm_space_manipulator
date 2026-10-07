"""Private state-local envelope with fixed arc indexing and full checks."""

from __future__ import annotations

import numpy as np

from v6_lite.pcc_fixed_arc_positions import FixedArcPCCPositions
from v6_lite.pcc_state_local_envelope import (
    ARITHMETIC_PAD_M, PCC_SAMPLES_PER_SEGMENT,
    CapsuleContainment, StateLocalEnvelopeResult,
)
from v6_lite.pcc_vectorized_state_envelope import (
    VectorizedStateLocalPCCEnvelopeAudit,
)


class FixedArcStateLocalPCCEnvelopeAudit(
        VectorizedStateLocalPCCEnvelopeAudit):
    def __init__(self, model, spec) -> None:
        super().__init__(model, spec)
        self._positions = FixedArcPCCPositions(spec, self.arclengths)

    def evaluate(self, data, planner_q, base_transform) -> StateLocalEnvelopeResult:
        pcc = self._positions.evaluate(planner_q, base_transform).reshape(
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

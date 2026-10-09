"""Private PCC capsule envelope with squared-distance reduction.

The 5-by-17 axis-to-PCC distance checks, allowances, radii, and numerical
pad are unchanged. Monotone min/max reduction precedes the square root, so
only one root is needed per capsule instead of one per sampled point pair.
"""

from __future__ import annotations

import numpy as np

from v6_lite.b2_compact_fixed_arc_envelope import (
    CompactEnvelopeSummary, CompactFixedArcStateLocalPCCEnvelopeAudit,
)
from v6_lite.pcc_state_local_envelope import (
    ARITHMETIC_PAD_M, PCC_SAMPLES_PER_SEGMENT,
)


class SquaredDistanceFixedArcEnvelopeAudit(
        CompactFixedArcStateLocalPCCEnvelopeAudit):
    def evaluate(self, data, planner_q, base_transform) -> CompactEnvelopeSummary:
        pcc = self._positions.evaluate(planner_q, base_transform).reshape(
            len(self.spec.segment_lengths_m), PCC_SAMPLES_PER_SEGMENT, 3,
        )
        rotation = np.asarray(data.xmat[self._body_ids],
                              dtype=np.float64).reshape(-1, 3, 3)
        origin = np.asarray(data.xpos[self._body_ids], dtype=np.float64)
        axis = origin[:, None, :] + np.matmul(
            self._local_axis, np.transpose(rotation, (0, 2, 1)),
        )
        difference = axis[:, :, None, :] - pcc[self._segment_ids][:, None, :, :]
        squared = np.sum(difference * difference, axis=3)
        sampled_max = np.sqrt(np.max(np.min(squared, axis=2), axis=1))
        required = np.nextafter(
            sampled_max + self._allowance_m + self._radius_m
            + self._excess_m + ARITHMETIC_PAD_M, np.inf,
        )
        margins = self._declared_m - required
        minimum = float(np.min(margins))
        return CompactEnvelopeSummary(
            status=("COVERED_AT_THIS_STATE" if minimum >= 0.0
                    else "NOT_COVERED_AT_THIS_STATE"),
            min_margin_m=minimum,
            checked_capsule_count=len(self.envelopes.capsules),
        )

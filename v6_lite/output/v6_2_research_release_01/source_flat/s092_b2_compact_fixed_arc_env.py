"""Private envelope summary without unused per-capsule result objects.

All original capsule samples, PCC points, radii, allowances and margins are
computed. This class is only used by the private timing trial; production and
independent per-capsule audit continue to use the full result class.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from v6_lite.pcc_fixed_arc_state_envelope import FixedArcStateLocalPCCEnvelopeAudit
from v6_lite.pcc_state_local_envelope import (
    ARITHMETIC_PAD_M, PCC_SAMPLES_PER_SEGMENT,
)


@dataclass(frozen=True)
class CompactEnvelopeSummary:
    status: str
    min_margin_m: float
    checked_capsule_count: int


class CompactFixedArcStateLocalPCCEnvelopeAudit(
        FixedArcStateLocalPCCEnvelopeAudit):
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
        minimum = float(np.min(margins))
        return CompactEnvelopeSummary(
            status=("COVERED_AT_THIS_STATE" if minimum >= 0.0
                    else "NOT_COVERED_AT_THIS_STATE"),
            min_margin_m=minimum,
            checked_capsule_count=len(self.envelopes.capsules),
        )

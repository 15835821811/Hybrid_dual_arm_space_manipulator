"""Private state-local envelope evaluator with reused PCC section prefixes.

Only the no-Jacobian curve point calculation differs from the reference
StateLocalPCCEnvelopeAudit. The same fitted capsules, sample positions,
sampling allowance, arithmetic pad and declared tube radii are retained.
This class is not selected by the production controller.
"""

from __future__ import annotations

from v6_lite.pcc_batched_point_model import BatchedPointContinuumShapeModel
from v6_lite.pcc_state_local_envelope import StateLocalPCCEnvelopeAudit


class PreparedStateLocalPCCEnvelopeAudit(StateLocalPCCEnvelopeAudit):
    def __init__(self, model, spec) -> None:
        super().__init__(model, spec)
        self.shape = BatchedPointContinuumShapeModel(spec)

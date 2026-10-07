"""Private next-start row builder with conservative MuJoCo pair screening.

Bounding spheres can exclude only pairs whose distance lower bound is above
the unchanged activation distance plus the existing numerical screen pad.
Every retained pair uses the original exact MuJoCo distance/witness code;
all PCC and actual-chain capsule rows are built without alteration.
"""

from __future__ import annotations

from dataclasses import replace

import mujoco
import numpy as np

SPHERE_SCREEN_PAD_M = 1e-6
from v6_lite.b2_reused_next_start import ReusedNextStart
from v6_lite.b2_batched_capsule_bounds import BatchedCapsuleMidpointBounds
from v6_lite.pcc_batched_distance_query import BatchedDistanceDecisionQuery
from v6_lite.recompute_execution_constraints import ReplayConstraintBuilder


class ScreenedReplayConstraintBuilder(ReplayConstraintBuilder):
    def __init__(self, spec, model, pairs, config) -> None:
        super().__init__(spec, model, pairs, config)
        self._all_pairs = self.pairs
        self._a = np.asarray([item.geom_a for item in self._all_pairs],
                             dtype=np.int32)
        self._b = np.asarray([item.geom_b for item in self._all_pairs],
                             dtype=np.int32)
        self._radius_a = np.asarray(model.geom_rbound[self._a],
                                    dtype=np.float64)
        self._radius_b = np.asarray(model.geom_rbound[self._b],
                                    dtype=np.float64)
        self.calls = 0
        self.exact_pair_calls = 0
        self.minimum_retained_pairs = len(self._all_pairs)
        self._batched_capsules = BatchedCapsuleMidpointBounds(self.capsules)

    def _minimum_capsule(self, data, target_box):
        return self._batched_capsules.minimum(self.model, data, target_box)

    def build(self, data: mujoco.MjData):
        mujoco.mj_forward(self.model, data)
        return self.build_prepared(data)

    def build_prepared(self, data: mujoco.MjData):
        """Screen rows after the preview has forwarded the identical state."""
        center_a = np.asarray(data.geom_xpos[self._a], dtype=np.float64)
        center_b = np.asarray(data.geom_xpos[self._b], dtype=np.float64)
        finite = (
            np.isfinite(self._radius_a) & np.isfinite(self._radius_b)
            & (self._radius_a > 0.0) & (self._radius_b > 0.0)
            & np.all(np.isfinite(center_a), axis=1)
            & np.all(np.isfinite(center_b), axis=1)
        )
        lower = np.linalg.norm(center_a - center_b, axis=1)
        lower -= self._radius_a
        lower -= self._radius_b
        far = finite & (lower > self.config.clearance_activation_m
                        + SPHERE_SCREEN_PAD_M)
        retained = tuple(item for index, item in enumerate(self._all_pairs)
                         if not far[index])
        self.calls += 1
        self.exact_pair_calls += len(retained)
        self.minimum_retained_pairs = min(self.minimum_retained_pairs,
                                          len(retained))
        self.pairs = retained
        try:
            return super()._build_forwarded(data)
        finally:
            self.pairs = self._all_pairs


class ScreenedNextStart(ReusedNextStart):
    """Reuse one screened builder, keeping the original next-start logic."""

    instances: list["ScreenedNextStart"] = []

    def __init__(self, *, track_instances=True) -> None:
        super().__init__()
        if track_instances:
            self.instances.append(self)

    def initialize(self, model, robot, verifier, cfg, evaluator):
        """Allocate model-constant geometry before the first timed cycle."""
        if self._model is not model:
            no_legacy = replace(cfg, enable_pcc_cbf=False,
                                enable_capsule_cbf=True)
            self._builder = ScreenedReplayConstraintBuilder(
                robot, model, verifier.pairs, no_legacy)
            self._query = BatchedDistanceDecisionQuery(
                evaluator.shape_model)
            self._model = model
            self.builder_constructions += 1

    def __call__(self, model, robot, verifier, cfg, evaluator, data,
                 endpoint_command):
        self.initialize(model, robot, verifier, cfg, evaluator)
        return super().__call__(model, robot, verifier, cfg, evaluator,
                                data, endpoint_command)

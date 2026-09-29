"""Private exact-input reuse of a just-computed interval decision.

The previous cycle's realized next start is often the next cycle's preflight
state. Reuse is allowed only when every query input, budget, partition and
shape-model object agrees exactly. All misses call the unchanged query.
"""

from __future__ import annotations

from dataclasses import replace
import time

import numpy as np

from v6_lite.pcc_persistent_interval_query import PersistentIntervalQueryResult
from v6_lite.shape_clearance import OrientedBox


class SameStateDecisionCache:
    def __init__(self, original_evaluate) -> None:
        self.original_evaluate = original_evaluate
        self._last_key = None
        self._last_result: PersistentIntervalQueryResult | None = None
        self.hit_count = 0
        self.miss_count = 0
        self.hit_lookup_ms = 0.0
        self.mismatch_counts = {name: 0 for name in (
            "shape_model", "q", "base", "box_center", "box_rotation",
            "box_extents", "partition_or_budget")}
        self.maximum_input_difference = {name: 0.0 for name in (
            "q", "base", "box_center", "box_rotation", "box_extents")}

    @staticmethod
    def _snapshot(result: PersistentIntervalQueryResult) -> PersistentIntervalQueryResult:
        return replace(
            result,
            lower_by_interval_id=dict(result.lower_by_interval_id),
            midpoint_upper_by_interval_id=dict(
                result.midpoint_upper_by_interval_id),
        )

    @staticmethod
    def _key(shape_model, q, base_transform, target_box: OrientedBox,
             partition, gate_m, max_point_evaluations, max_leaves,
             numerical_pad_m):
        return (
            shape_model,
            np.asarray(q, dtype=np.float64).copy(),
            np.asarray(base_transform, dtype=np.float64).copy(),
            target_box.center.copy(), target_box.rotation.copy(),
            target_box.half_extents.copy(), partition.leaves,
            gate_m, max_point_evaluations, max_leaves, numerical_pad_m,
        )

    @staticmethod
    def _same(first, second) -> bool:
        if first is None or first[0] is not second[0]:
            return False
        return (all(np.array_equal(first[index], second[index])
                    for index in range(1, 6))
                and first[6:] == second[6:])

    def _record_mismatch(self, key) -> None:
        previous = self._last_key
        if previous is None:
            return
        if previous[0] is not key[0]:
            self.mismatch_counts["shape_model"] += 1
        for index, name in enumerate(("q", "base", "box_center",
                                      "box_rotation", "box_extents"), 1):
            if not np.array_equal(previous[index], key[index]):
                self.mismatch_counts[name] += 1
                if previous[index].shape == key[index].shape:
                    difference = float(np.max(np.abs(
                        previous[index] - key[index])))
                    self.maximum_input_difference[name] = max(
                        self.maximum_input_difference[name], difference)
        if previous[6:] != key[6:]:
            self.mismatch_counts["partition_or_budget"] += 1

    def evaluate(self, query, q, base_transform, target_box, partition,
                 *, gate_m, max_point_evaluations, max_leaves,
                 numerical_pad_m) -> PersistentIntervalQueryResult:
        started = time.perf_counter()
        key = self._key(query.shape_model, q, base_transform, target_box,
                        partition, gate_m, max_point_evaluations, max_leaves,
                        numerical_pad_m)
        if self._last_result is not None and self._same(self._last_key, key):
            self.hit_count += 1
            result = self._snapshot(self._last_result)
            self.hit_lookup_ms += (time.perf_counter() - started) * 1000.0
            return result
        self._record_mismatch(key)
        self.miss_count += 1
        result = self.original_evaluate(
            query, q, base_transform, target_box, partition,
            gate_m=gate_m, max_point_evaluations=max_point_evaluations,
            max_leaves=max_leaves, numerical_pad_m=numerical_pad_m)
        if result.bounds_valid and not result.budget_exhausted:
            self._last_key = key
            self._last_result = self._snapshot(result)
        else:
            self._last_key = None
            self._last_result = None
        return result

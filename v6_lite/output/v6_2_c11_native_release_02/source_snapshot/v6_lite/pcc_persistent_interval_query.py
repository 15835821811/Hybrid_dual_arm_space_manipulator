"""Decision-oriented PCC bound query with a persistent material partition.

Every call reevaluates all current leaf midpoints at the *current* state. It
may only split leaves at the planning boundary; unchanged leaves reuse their
same-state values during the refinement loop. No merge or fixed top-k drop is
performed. This is a read-only B.2 shadow component until online admission.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from v6_lite.continuum_shape_model import ContinuumShapeModel, validate_transform
from v6_lite.pcc_clearance import V61A_PCC_TUBE_RADII_M
from v6_lite.pcc_interval_cbf import IntervalPartition, MaterialInterval, NUMERICAL_PAD_M
from v6_lite.shape_clearance import OrientedBox, point_obb_signed_distance


@dataclass(frozen=True)
class PersistentIntervalQueryResult:
    partition: IntervalPartition
    distance_lower_bound_m: float
    distance_upper_bound_m: float
    bound_gap_m: float
    bounds_valid: bool
    proxy_clearance_status: str
    budget_exhausted: bool
    failure_reason: str | None
    point_evaluation_count: int
    split_count: int
    interval_count: int
    elapsed_ms: float
    lower_by_interval_id: dict[str, float]
    midpoint_upper_by_interval_id: dict[str, float]


class PersistentIntervalDecisionQuery:
    def __init__(self, shape_model: ContinuumShapeModel | None = None) -> None:
        self.shape_model = shape_model or ContinuumShapeModel()

    def evaluate(self, q: np.ndarray, base_transform: np.ndarray,
                 target_box: OrientedBox, partition: IntervalPartition,
                 *, gate_m: float = 0.005,
                 max_point_evaluations: int = 64,
                 max_leaves: int = 128,
                 numerical_pad_m: float = NUMERICAL_PAD_M) -> PersistentIntervalQueryResult:
        started = time.perf_counter()
        q = np.asarray(q, dtype=np.float64)
        if q.shape != (10,) or np.any(~np.isfinite(q)):
            raise ValueError("finite 10D planner configuration required")
        base = validate_transform(base_transform)
        if not math.isfinite(gate_m) or not math.isfinite(numerical_pad_m):
            raise ValueError("gate and numerical pad must be finite")
        if numerical_pad_m < 0 or max_point_evaluations < 5 or max_leaves < 5:
            raise ValueError("invalid query budget or numerical pad")
        lengths = self.shape_model.spec.segment_lengths_m
        coverage = partition.coverage(lengths)
        if not coverage.coverage_complete:
            return PersistentIntervalQueryResult(
                partition, math.nan, math.nan, math.nan, False,
                "UNKNOWN_INVALID_PARTITION", False, "INCOMPLETE_COVERAGE",
                0, 0, len(partition.leaves),
                (time.perf_counter() - started) * 1000.0, {}, {},
            )
        if len(partition.leaves) > max_point_evaluations or len(partition.leaves) > max_leaves:
            return PersistentIntervalQueryResult(
                partition, math.nan, math.nan, math.nan, False,
                "UNKNOWN_BUDGET_INSUFFICIENT", True,
                "BUDGET_CANNOT_EVALUATE_ALL_EXISTING_LEAVES",
                0, 0, len(partition.leaves),
                (time.perf_counter() - started) * 1000.0, {}, {},
            )
        boundaries = self.shape_model.spec.segment_boundaries_m
        leaves = tuple(sorted(partition.leaves))
        point_count = 0
        split_count = 0
        values: dict[str, tuple[float, float]] = {}

        def evaluate_leaves(items: tuple[MaterialInterval, ...]) -> None:
            nonlocal point_count
            arclengths = []
            for leaf in items:
                a, b = leaf.local_bounds(float(lengths[leaf.segment_id]))
                arclengths.append(float(boundaries[leaf.segment_id] + 0.5 * (a + b)))
            points = self.shape_model.batch_query(q, base, arclengths,
                                                  with_jacobians=False)
            for leaf, point in zip(items, points):
                a, b = leaf.local_bounds(float(lengths[leaf.segment_id]))
                upper = (point_obb_signed_distance(point.position_world,
                                                   target_box).signed_distance_m
                         - float(V61A_PCC_TUBE_RADII_M[leaf.segment_id]))
                lower = upper - 0.5 * (b - a) - numerical_pad_m
                if not math.isfinite(lower) or not math.isfinite(upper):
                    raise ValueError("nonfinite PCC interval distance")
                values[leaf.interval_id] = (float(lower), float(upper))
            point_count += len(items)

        evaluate_leaves(leaves)
        failure = None
        budget_exhausted = False
        while True:
            lower = min(item[0] for item in values.values())
            upper = min(item[1] for item in values.values())
            if lower >= gate_m or upper < gate_m:
                break
            if point_count + 2 > max_point_evaluations or len(leaves) + 1 > max_leaves:
                budget_exhausted = True
                failure = "DECISION_UNKNOWN_AT_BUDGET"
                break
            worst = min(leaves, key=lambda item: (values[item.interval_id][0],
                                                 item.interval_id))
            a, b = worst.local_bounds(float(lengths[worst.segment_id]))
            if a + (b - a) / 2.0 in (a, b):
                failure = "FLOATING_POINT_SUBDIVISION_STALLED"
                break
            left, right = worst.children()
            del values[worst.interval_id]
            evaluate_leaves((left, right))
            leaves = tuple(sorted([x for x in leaves if x != worst] + [left, right]))
            split_count += 1
        raw_lower = min(item[0] for item in values.values())
        raw_upper = min(item[1] for item in values.values())
        lower = float(np.nextafter(raw_lower, -math.inf))
        upper = float(np.nextafter(raw_upper + numerical_pad_m, math.inf))
        if lower >= gate_m:
            status = "PROXY_CLEARANCE_AT_LEAST_GATE"
        elif upper < gate_m:
            status = "PROXY_CLEARANCE_BELOW_GATE"
        else:
            status = "UNKNOWN_CROSSES_GATE"
        updated = IntervalPartition(leaves)
        return PersistentIntervalQueryResult(
            updated, lower, upper, upper - lower, True, status,
            budget_exhausted, failure, point_count, split_count,
            len(leaves), (time.perf_counter() - started) * 1000.0,
            {key: item[0] for key, item in values.items()},
            {key: item[1] for key, item in values.items()},
        )

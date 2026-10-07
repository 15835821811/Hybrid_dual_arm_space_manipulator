"""Read-only same-state PCC prefix reuse trial for the persistent query.

The decision algorithm and fixed interval function mirror the frozen reference
query. Full-section transforms are prepared once for each state, and a long
leaf batch uses vectorized midpoint-to-OBB distances. This implementation is
not selected by the online controller.
"""

from __future__ import annotations

import math
import time

import numpy as np

from v6_lite.continuum_shape_model import ContinuumShapeModel, validate_transform
from v6_lite.pcc_clearance import V61A_PCC_TUBE_RADII_M
from v6_lite.pcc_interval_cbf import IntervalPartition, MaterialInterval, NUMERICAL_PAD_M
from v6_lite.pcc_persistent_interval_query import PersistentIntervalQueryResult
from v6_lite.shape_clearance import OrientedBox, point_obb_signed_distance


def signed_distance_many(points: np.ndarray, target: OrientedBox) -> np.ndarray:
    """Match point_obb_signed_distance, including its interior branch."""

    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("point batch must have shape (N,3)")
    local = (values - target.center) @ target.rotation
    closest = np.clip(local, -target.half_extents, target.half_extents)
    exterior = np.linalg.norm(local - closest, axis=1)
    interior = -np.min(target.half_extents - np.abs(local), axis=1)
    return np.where(exterior > 1e-14, exterior, interior)


class _PreparedPCCPoints:
    def __init__(self, shape: ContinuumShapeModel, q: np.ndarray,
                 base: np.ndarray) -> None:
        self.shape = shape
        self.q = q
        self.prefixes = [base @ shape.spec.base_to_shape_start]
        for segment_id, length in enumerate(shape.spec.segment_lengths_m):
            full = shape._section_transform(
                q[2*segment_id:2*segment_id+2], float(length),
                float(length), shape.spec.pcc_bending_map,
            )
            self.prefixes.append(self.prefixes[-1] @ full)

    def positions(self, items: tuple[MaterialInterval, ...],
                  bounds: list[tuple[float, float]]) -> np.ndarray:
        positions = np.empty((len(items), 3), dtype=np.float64)
        for index, (leaf, (start, end)) in enumerate(zip(items, bounds)):
            segment_id = leaf.segment_id
            length = float(self.shape.spec.segment_lengths_m[segment_id])
            local = self.shape._section_transform(
                self.q[2*segment_id:2*segment_id+2], length,
                .5 * (start + end), self.shape.spec.pcc_bending_map,
            )
            transform = self.prefixes[segment_id] @ local
            positions[index] = transform[:3, 3]
        return positions


class BatchedDistanceDecisionQuery:
    def __init__(self, shape_model: ContinuumShapeModel | None = None) -> None:
        self.shape_model = shape_model or ContinuumShapeModel()

    def evaluate(self, q: np.ndarray, base_transform: np.ndarray,
                 target_box: OrientedBox, partition: IntervalPartition,
                 *, gate_m: float = .005,
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
        leaves = tuple(sorted(partition.leaves))
        prepared = _PreparedPCCPoints(self.shape_model, q, base)
        point_count = 0
        split_count = 0
        values: dict[str, tuple[float, float]] = {}

        def evaluate_leaves(items: tuple[MaterialInterval, ...]) -> None:
            nonlocal point_count
            local_bounds = [leaf.local_bounds(float(lengths[leaf.segment_id]))
                            for leaf in items]
            points = prepared.positions(items, local_bounds)
            signed = (
                signed_distance_many(points, target_box)
                if len(points) >= 8 else
                np.asarray([point_obb_signed_distance(
                    point, target_box).signed_distance_m
                    for point in points], dtype=np.float64)
            )
            for leaf, (a, b), value in zip(items, local_bounds, signed):
                upper = float(value - V61A_PCC_TUBE_RADII_M[leaf.segment_id])
                lower = upper - .5 * (b - a) - numerical_pad_m
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
            if a + (b-a)/2.0 in (a, b):
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
            updated, lower, upper, upper-lower, True, status,
            budget_exhausted, failure, point_count, split_count,
            len(leaves), (time.perf_counter()-started)*1000.0,
            {key: item[0] for key, item in values.items()},
            {key: item[1] for key, item in values.items()},
        )

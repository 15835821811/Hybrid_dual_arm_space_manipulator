"""Offline PCC tube/OBB distance bounds; never used by the online CBF.

For each fixed-shape section, the centerline is parameterized by material
arclength. Its derivative is R(s)e_x, so its norm is one in exact arithmetic.
Signed distance to a closed OBB is 1-Lipschitz in the point. Thus a midpoint
sample over [a,b] gives f(mid) - (b-a)/2 <= min f <= f(mid). Every section is
covered separately because the calibrated tube radii differ by section.

Outward ``nextafter`` and a configurable 1 nm pad protect against ordinary
double-precision rounding in tested cases. This is not an interval-arithmetic
or formally certified floating-point implementation. A valid interval bounds
the defined PCC proxy under the stated exact-model assumptions; it does not
prove the proxy envelops the physical robot throughout the work domain.
"""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import minimize_scalar

from v6_lite.continuum_shape_model import ContinuumShapeModel, validate_transform
from v6_lite.pcc_clearance import V61A_PCC_TUBE_RADII_M
from v6_lite.shape_clearance import OrientedBox, point_obb_signed_distance


@dataclass(frozen=True)
class CurveSegment:
    length_m: float
    radius_m: float
    point_at_local_s: Callable[[float], np.ndarray]
    lipschitz_m_per_m: float = 1.0


@dataclass(frozen=True)
class CandidateInterval:
    segment_id: int
    local_start_m: float
    local_end_m: float
    lower_bound_m: float
    midpoint_distance_m: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "segment_id": self.segment_id,
            "local_start_m": self.local_start_m,
            "local_end_m": self.local_end_m,
            "lower_bound_m": self.lower_bound_m,
            "midpoint_distance_m": self.midpoint_distance_m,
        }


@dataclass(frozen=True)
class BoundedClearanceResult:
    distance_lower_bound_m: float
    distance_upper_bound_m: float
    bound_gap_m: float
    bounds_valid: bool
    tolerance_met: bool
    budget_exhausted: bool
    proxy_clearance_status: str
    geometry_domain_status: str
    envelope_status: str
    numerical_status: str
    candidate_intervals: tuple[CandidateInterval, ...]
    evaluation_count: int
    interval_count: int
    local_refinement_evaluation_count: int
    elapsed_ms: float
    best_segment_id: int
    best_local_arclength_m: float
    best_point_world: np.ndarray
    projected_subspace_residual_linf_rad: float | None = None

    def to_dict(self, *, include_intervals: bool = True) -> dict:
        result = {
            "distance_lower_bound_m": self.distance_lower_bound_m,
            "distance_upper_bound_m": self.distance_upper_bound_m,
            "bound_gap_m": self.bound_gap_m,
            "bounds_valid": self.bounds_valid,
            "tolerance_met": self.tolerance_met,
            "budget_exhausted": self.budget_exhausted,
            "proxy_clearance_status": self.proxy_clearance_status,
            "geometry_domain_status": self.geometry_domain_status,
            "envelope_status": self.envelope_status,
            "numerical_status": self.numerical_status,
            "evaluation_count": self.evaluation_count,
            "interval_count": self.interval_count,
            "local_refinement_evaluation_count": self.local_refinement_evaluation_count,
            "elapsed_ms": self.elapsed_ms,
            "best_segment_id": self.best_segment_id,
            "best_local_arclength_m": self.best_local_arclength_m,
            "best_point_world": self.best_point_world.tolist(),
            "projected_subspace_residual_linf_rad": self.projected_subspace_residual_linf_rad,
        }
        if include_intervals:
            result["candidate_intervals"] = [x.to_dict() for x in self.candidate_intervals]
        return result


def bounded_curve_obb_clearance(
    segments: tuple[CurveSegment, ...],
    target_obb: OrientedBox,
    *,
    safety_gate_m: float = 0.005,
    tolerance_m: float = 0.001,
    max_evaluations: int = 101,
    numerical_pad_m: float = 1e-9,
    local_refinement_count: int = 0,
    geometry_domain_status: str = "NOT_ASSESSED",
    envelope_status: str = "FINITE_REGRESSION_ONLY",
    projected_subspace_residual_linf_rad: float | None = None,
) -> BoundedClearanceResult:
    """Cover every segment and return a global interval for the proxy minimum.

    Custom curves are intended for analytic tests; callers must supply a true
    Lipschitz bound. The PCC wrapper derives its K=1 from the section model.
    Local minimization only changes the sampled upper bound.
    """
    if not segments or max_evaluations < len(segments):
        raise ValueError("evaluation budget must cover every segment midpoint")
    if not all(math.isfinite(x) for x in (safety_gate_m, tolerance_m, numerical_pad_m)):
        raise ValueError("query thresholds must be finite")
    if tolerance_m <= 0.0 or numerical_pad_m < 0.0 or local_refinement_count < 0:
        raise ValueError("invalid query tolerance, pad, or refinement count")
    for segment in segments:
        if not all(math.isfinite(x) for x in (
            segment.length_m, segment.radius_m, segment.lipschitz_m_per_m
        )) or segment.length_m <= 0.0 or segment.radius_m < 0.0 or segment.lipschitz_m_per_m <= 0.0:
            raise ValueError("invalid curve segment or Lipschitz bound")

    started = time.perf_counter()
    evaluation_count = 0
    local_count = 0
    best_upper = math.inf
    best_segment = -1
    best_s = math.nan
    best_point = np.full(3, math.nan)
    heap: list[tuple[float, int, CandidateInterval]] = []
    sequence = 0

    def sample(segment_id: int, local_s: float, *, local: bool = False) -> float:
        nonlocal evaluation_count, local_count, best_upper, best_segment, best_s, best_point
        point = np.asarray(segments[segment_id].point_at_local_s(local_s), dtype=np.float64)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise ValueError("curve evaluator returned an invalid point")
        distance = float(point_obb_signed_distance(point, target_obb).signed_distance_m
                         - segments[segment_id].radius_m)
        if not math.isfinite(distance):
            raise ValueError("nonfinite point-to-OBB distance")
        if local:
            local_count += 1
        else:
            evaluation_count += 1
        if distance < best_upper:
            best_upper = distance
            best_segment, best_s, best_point = segment_id, local_s, point.copy()
        return distance

    def interval(segment_id: int, a: float, b: float) -> CandidateInterval:
        mid = a + (b - a) / 2.0
        value = sample(segment_id, mid)
        lower = np.nextafter(
            value - segments[segment_id].lipschitz_m_per_m * (b - a) / 2.0
            - numerical_pad_m, -math.inf
        )
        return CandidateInterval(segment_id, a, b, float(lower), value)

    for i, segment in enumerate(segments):
        item = interval(i, 0.0, segment.length_m)
        heapq.heappush(heap, (item.lower_bound_m, sequence, item))
        sequence += 1

    # Splitting two children requires two evaluations. Keep every leaf until
    # its lower bound cannot beat the best sampled upper bound.
    while heap and evaluation_count + 2 <= max_evaluations:
        global_lower = min(best_upper, heap[0][0])
        if best_upper - global_lower + 2 * numerical_pad_m <= tolerance_m:
            break
        _, _, worst = heapq.heappop(heap)
        if worst.lower_bound_m >= best_upper:
            # Every remaining leaf also has a lower bound >= best_upper.
            # Retain all leaves so the returned partition visibly covers
            # every original section even after proof-based early stopping.
            heapq.heappush(heap, (worst.lower_bound_m, sequence, worst))
            sequence += 1
            break
        mid = worst.local_start_m + (worst.local_end_m - worst.local_start_m) / 2.0
        if mid <= worst.local_start_m or mid >= worst.local_end_m:
            heapq.heappush(heap, (worst.lower_bound_m, sequence, worst))
            sequence += 1
            break
        for a, b in ((worst.local_start_m, mid), (mid, worst.local_end_m)):
            child = interval(worst.segment_id, a, b)
            heapq.heappush(heap, (child.lower_bound_m, sequence, child))
            sequence += 1

    # Optional local search is never a source of a lower bound. It may improve
    # the upper bound even when the branch budget is exhausted.
    for _, _, item in heapq.nsmallest(local_refinement_count, heap):
        minimize_scalar(
            lambda s, i=item.segment_id: sample(i, float(s), local=True),
            method="bounded", bounds=(item.local_start_m, item.local_end_m),
            options={"maxiter": 16, "xatol": 1e-7},
        )

    candidates = tuple(item for _, _, item in sorted(heap))
    raw_lower = min([best_upper, *(item.lower_bound_m for item in candidates)])
    lower = float(np.nextafter(raw_lower, -math.inf))
    upper = float(np.nextafter(best_upper + numerical_pad_m, math.inf))
    gap = max(0.0, upper - lower)
    tolerance_met = gap <= tolerance_m
    budget_exhausted = not tolerance_met and evaluation_count + 2 > max_evaluations
    if lower >= safety_gate_m:
        proxy_status = "PROXY_CLEARANCE_AT_LEAST_GATE"
    elif upper < safety_gate_m:
        proxy_status = "PROXY_CLEARANCE_BELOW_GATE"
    else:
        proxy_status = "UNKNOWN_CROSSES_GATE"
    return BoundedClearanceResult(
        distance_lower_bound_m=lower,
        distance_upper_bound_m=upper,
        bound_gap_m=gap,
        bounds_valid=math.isfinite(lower) and math.isfinite(upper) and lower <= upper,
        tolerance_met=tolerance_met,
        budget_exhausted=budget_exhausted,
        proxy_clearance_status=proxy_status,
        geometry_domain_status=geometry_domain_status,
        envelope_status=envelope_status,
        numerical_status="OUTWARD_PAD_EMPIRICAL_NOT_INTERVAL_CERTIFIED",
        candidate_intervals=candidates,
        evaluation_count=evaluation_count,
        interval_count=len(candidates),
        local_refinement_evaluation_count=local_count,
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
        best_segment_id=best_segment,
        best_local_arclength_m=best_s,
        best_point_world=best_point,
        projected_subspace_residual_linf_rad=projected_subspace_residual_linf_rad,
    )


class PCCBoundedClearanceEvaluator:
    """Five-section read-only query using the existing PCC radii and model."""

    def __init__(self, shape_model: ContinuumShapeModel | None = None) -> None:
        self.shape_model = ContinuumShapeModel() if shape_model is None else shape_model

    def evaluate(
        self,
        q_continuum: np.ndarray,
        base_transform: np.ndarray,
        target_obb: OrientedBox,
        *,
        actual_configuration: np.ndarray | None = None,
        **query_options,
    ) -> BoundedClearanceResult:
        spec = self.shape_model.spec
        q = np.asarray(q_continuum, dtype=np.float64)
        if q.shape != (10,) or not np.all(np.isfinite(q)):
            raise ValueError("planner configuration must be finite with shape (10,)")
        base = validate_transform(base_transform)
        residual = None
        if actual_configuration is not None:
            projection = spec.project_actual_configuration(actual_configuration)
            residual = projection.residual_linf_rad
            # This is a membership check for the declared 10-to-60 map, not
            # a tolerance relaxation of any online collision constraint.
            subspace_status = ("ON_DECLARED_SHAPE_SUBSPACE" if residual <= 1e-10
                               and np.max(np.abs(projection.planner_configuration - q)) <= 1e-10
                               else "OUTSIDE_DECLARED_SHAPE_SUBSPACE")
        else:
            subspace_status = "ACTUAL_CHAIN_NOT_PROVIDED"
        in_domain = bool(np.all(q >= spec.work_domain_lower_rad)
                         and np.all(q <= spec.work_domain_upper_rad))
        domain = ("INSIDE_DECLARED_WORK_DOMAIN" if in_domain
                  else "OUTSIDE_DECLARED_WORK_DOMAIN") + ";" + subspace_status
        envelope = ("FINITE_V6_1A_REGRESSION_ONLY" if in_domain
                    and subspace_status != "OUTSIDE_DECLARED_SHAPE_SUBSPACE"
                    else "NO_ENVELOPE_EVIDENCE_FOR_THIS_STATE")
        boundaries = spec.segment_boundaries_m
        segments = tuple(
            CurveSegment(
                length_m=float(length), radius_m=float(V61A_PCC_TUBE_RADII_M[i]),
                point_at_local_s=lambda s, i=i: self.shape_model.evaluate(
                    q, base, float(boundaries[i] + s), with_jacobians=False
                ).position_world,
                lipschitz_m_per_m=1.0,
            )
            for i, length in enumerate(spec.segment_lengths_m)
        )
        return bounded_curve_obb_clearance(
            segments, target_obb, geometry_domain_status=domain,
            envelope_status=envelope,
            projected_subspace_residual_linf_rad=residual,
            **query_options,
        )

"""Online PCC-tube clearance against the moving target satellite OBB.

For each PCC section the safety quantity is

``d_i(s) = sd_OBB(p_i(s)) - r_i``.

The global clearance is the minimum over section and arclength.  A small,
fixed coarse grid supplies a broad phase; only the most promising sections
are locally refined.  At a smooth closest feature the shape gradient follows
the envelope theorem:

``d d / d q_c = n_box_to_arm @ d p(s*) / d q_c``.

The calibrated radii include the physical envelope, PCC/discrete-chain model
discrepancy, arclength sampling allowance, and the V6.1-A numerical margin.
They are conservative proxy radii, not physical arm radii.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from v6_lite.continuum_shape_model import ContinuumShapeModel
from v6_lite.shape_clearance import OrientedBox, point_obb_signed_distance

V61A_PCC_TUBE_RADII_M = np.asarray(
    [
        0.0683669705070089,
        0.0850488419789490,
        0.0884648586864696,
        0.0925342712198274,
        0.0884178929464368,
    ],
    dtype=np.float64,
)
V61A_PCC_TUBE_RADII_M.setflags(write=False)


@dataclass(frozen=True)
class PCCClearanceResult:
    """Closest PCC-tube/OBB feature and its fixed-base shape gradient."""

    distance: float
    segment_id: int
    arc_length: float
    closest_point: np.ndarray
    point_on_obb: np.ndarray
    normal: np.ndarray
    radius: float
    gradient: np.ndarray
    centerline_point: np.ndarray
    coarse_evaluation_count: int
    refinement_evaluation_count: int
    refined_segment_count: int

    @property
    def signed_distance_m(self) -> float:
        return self.distance


class PCCClearanceEvaluator:
    """Deterministic broad-phase plus local-refinement PCC distance query."""

    def __init__(
        self,
        shape_model: ContinuumShapeModel | None = None,
        *,
        tube_radii_m: np.ndarray = V61A_PCC_TUBE_RADII_M,
        coarse_samples_per_segment: int = 7,
        active_segment_count: int = 2,
        refinement_max_iterations: int = 12,
        refinement_tolerance_m: float = 1e-5,
    ) -> None:
        self.shape_model = (
            ContinuumShapeModel() if shape_model is None else shape_model
        )
        self.tube_radii_m = np.asarray(tube_radii_m, dtype=np.float64).copy()
        self.coarse_samples_per_segment = int(coarse_samples_per_segment)
        self.active_segment_count = int(active_segment_count)
        self.refinement_max_iterations = int(refinement_max_iterations)
        self.refinement_tolerance_m = float(refinement_tolerance_m)
        if self.tube_radii_m.shape != (5,) or np.any(
            ~np.isfinite(self.tube_radii_m)
        ):
            raise ValueError("PCC tube radii must be finite with shape (5,)")
        if np.any(self.tube_radii_m <= 0.0):
            raise ValueError("PCC tube radii must be positive")
        if self.coarse_samples_per_segment < 3:
            raise ValueError("PCC broad phase needs at least three samples per segment")
        if self.active_segment_count not in range(1, 6):
            raise ValueError("active PCC segment count must lie in [1,5]")
        if self.refinement_max_iterations < 1:
            raise ValueError("PCC refinement iteration limit must be positive")
        if self.refinement_tolerance_m <= 0.0:
            raise ValueError("PCC refinement tolerance must be positive")

    def evaluate(
        self,
        q_continuum: np.ndarray,
        base_transform: np.ndarray,
        target_obb: OrientedBox,
    ) -> PCCClearanceResult:
        """Return the minimum finite-radius PCC clearance and shape gradient."""

        boundaries = self.shape_model.spec.segment_boundaries_m
        coarse_records: list[tuple[float, int, float, int]] = []
        segment_records: list[list[tuple[float, float, int]]] = []
        segment_grids = [
            np.linspace(
                0.0,
                float(length),
                self.coarse_samples_per_segment,
            )
            for length in self.shape_model.spec.segment_lengths_m
        ]
        all_arclengths = [
            float(boundaries[segment_index] + local)
            for segment_index, local_grid in enumerate(segment_grids)
            for local in local_grid
        ]
        all_points = self.shape_model.batch_query(
            q_continuum,
            base_transform,
            all_arclengths,
            with_jacobians=False,
        )
        evaluation_index = 0
        for segment_index, local_grid in enumerate(segment_grids):
            point_start = segment_index * self.coarse_samples_per_segment
            points = all_points[
                point_start : point_start + self.coarse_samples_per_segment
            ]
            current: list[tuple[float, float, int]] = []
            for local, point in zip(local_grid, points):
                signed = point_obb_signed_distance(point.position_world, target_obb)
                distance = float(
                    signed.signed_distance_m - self.tube_radii_m[segment_index]
                )
                record = (distance, float(local), evaluation_index)
                current.append(record)
                coarse_records.append(
                    (distance, segment_index, float(local), evaluation_index)
                )
                evaluation_index += 1
            segment_records.append(current)

        best_by_segment = [min(records, key=lambda item: item[0]) for records in segment_records]
        active_segments = np.argsort(
            np.asarray([item[0] for item in best_by_segment], dtype=np.float64)
        )[: self.active_segment_count]
        candidates = list(coarse_records)
        refinement_evaluations = 0
        for raw_segment_index in active_segments:
            segment_index = int(raw_segment_index)
            records = segment_records[segment_index]
            best_index = int(np.argmin([item[0] for item in records]))
            lower_index = max(0, best_index - 1)
            upper_index = min(len(records) - 1, best_index + 1)
            lower = float(records[lower_index][1])
            upper = float(records[upper_index][1])
            if upper - lower <= self.refinement_tolerance_m:
                continue

            def objective(
                local_arclength: float,
                active_segment_index: int = segment_index,
            ) -> float:
                nonlocal refinement_evaluations
                refinement_evaluations += 1
                arclength = float(
                    boundaries[active_segment_index] + local_arclength
                )
                point = self.shape_model.evaluate(
                    q_continuum,
                    base_transform,
                    arclength,
                    with_jacobians=False,
                )
                return float(
                    point_obb_signed_distance(
                        point.position_world, target_obb
                    ).signed_distance_m
                    - self.tube_radii_m[active_segment_index]
                )

            refined = minimize_scalar(
                objective,
                method="bounded",
                bounds=(lower, upper),
                options={
                    "xatol": self.refinement_tolerance_m,
                    "maxiter": self.refinement_max_iterations,
                },
            )
            candidates.append(
                (
                    float(refined.fun),
                    segment_index,
                    float(refined.x),
                    evaluation_index,
                )
            )
            evaluation_index += 1

        distance, segment_index, local_arclength, _ = min(
            candidates, key=lambda item: item[0]
        )
        arclength = float(boundaries[segment_index] + local_arclength)
        point = self.shape_model.evaluate(
            q_continuum,
            base_transform,
            arclength,
            with_jacobians=True,
        )
        signed = point_obb_signed_distance(point.position_world, target_obb)
        radius = float(self.tube_radii_m[segment_index])
        normal = np.asarray(signed.normal_box_to_point, dtype=np.float64)
        closest_point = point.position_world - radius * normal
        gradient = normal @ point.position_jacobian
        return PCCClearanceResult(
            distance=float(signed.signed_distance_m - radius),
            segment_id=int(segment_index),
            arc_length=arclength,
            closest_point=np.asarray(closest_point, dtype=np.float64),
            point_on_obb=np.asarray(signed.box_point, dtype=np.float64),
            normal=normal,
            radius=radius,
            gradient=np.asarray(gradient, dtype=np.float64),
            centerline_point=np.asarray(point.position_world, dtype=np.float64),
            coarse_evaluation_count=5 * self.coarse_samples_per_segment,
            refinement_evaluation_count=int(refinement_evaluations),
            refined_segment_count=len(active_segments),
        )


def compute_pcc_obb_clearance(
    q_continuum: np.ndarray,
    base_transform: np.ndarray,
    target_obb: OrientedBox,
    *,
    evaluator: PCCClearanceEvaluator | None = None,
) -> PCCClearanceResult:
    """Convenience function for a single PCC/OBB query."""

    active_evaluator = PCCClearanceEvaluator() if evaluator is None else evaluator
    return active_evaluator.evaluate(q_continuum, base_transform, target_obb)

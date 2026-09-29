"""Fixed-interval PCC safety functions and their matching differential rows.

This module is read-only in B.2-1. An interval's function is exactly
    h = sd_OBB(p(mid)) - radius - width/2 - numeric_pad - safe_distance.
The partition is fixed while differentiating: midpoint, width, tube radius,
and pad are constants. The signed-distance normal belongs to *that* midpoint
and its own OBB witness. We never differentiate B.1's branch decisions or
attach the old global witness gradient to an interval lower bound.

For the five-section PCC model, material arclength satisfies |dp/ds|=1 in
exact arithmetic. A complete partition and h>=0 on every leaf imply the
defined PCC proxy is above the gate under this analytic assumption. Floating
point outward rounding and real-chain envelope coverage remain separate
evidence questions, not hidden inside a single ``bounds_valid`` flag.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Sequence

import mujoco
import numpy as np

from model_test.robot_model_spec_v5 import RobotModelSpecV5
from v6_lite.continuum_model_spec import CONTINUUM_ACTUATED_DOF, default_continuum_model_spec
from v6_lite.continuum_shape_model import ContinuumShapeModel, transform_from_free_qpos
from v6_lite.hierarchical_qp import free_joint_slices, joint_addresses
from v6_lite.pcc_clearance import V61A_PCC_TUBE_RADII_M
from v6_lite.shape_clearance import OrientedBox, point_obb_signed_distance, target_box_from_mujoco


NUMERICAL_PAD_M = 1e-9
SHAPE_SUBSPACE_MEMBERSHIP_TOL_RAD = 1e-10
FEATURE_TIE_TOL_M = 1e-8


@dataclass(frozen=True, order=True)
class MaterialInterval:
    segment_id: int
    binary_path: str = ""

    def __post_init__(self) -> None:
        if self.segment_id not in range(5) or any(x not in "01" for x in self.binary_path):
            raise ValueError("interval requires segment 0..4 and a binary path")

    @property
    def interval_id(self) -> str:
        return f"pcc_interval:{self.segment_id}:{self.binary_path or 'root'}"

    def local_bounds(self, segment_length_m: float) -> tuple[float, float]:
        denominator = 1 << len(self.binary_path)
        ordinal = int(self.binary_path, 2) if self.binary_path else 0
        return (segment_length_m * ordinal / denominator,
                segment_length_m * (ordinal + 1) / denominator)

    def children(self) -> tuple[MaterialInterval, MaterialInterval]:
        return (MaterialInterval(self.segment_id, self.binary_path + "0"),
                MaterialInterval(self.segment_id, self.binary_path + "1"))


@dataclass(frozen=True)
class PartitionCoverage:
    interval_well_formed: bool
    coverage_complete: bool
    failure_reasons: tuple[str, ...]
    interval_count: int


@dataclass(frozen=True)
class IntervalPartition:
    leaves: tuple[MaterialInterval, ...]

    @classmethod
    def uniform(cls, depth: int = 0) -> IntervalPartition:
        if depth < 0 or depth > 20:
            raise ValueError("partition depth must lie in 0..20")
        return cls(tuple(MaterialInterval(i, format(j, f"0{depth}b") if depth else "")
                   for i in range(5) for j in range(1 << depth)))

    def refine(self, interval_ids: Sequence[str]) -> IntervalPartition:
        wanted = set(interval_ids)
        existing = {x.interval_id for x in self.leaves}
        if not wanted <= existing:
            raise ValueError("cannot refine a leaf absent from the partition")
        leaves = []
        for item in self.leaves:
            leaves.extend(item.children() if item.interval_id in wanted else (item,))
        return IntervalPartition(tuple(sorted(leaves)))

    def merge(self, parent: MaterialInterval) -> IntervalPartition:
        left, right = parent.children()
        if left not in self.leaves or right not in self.leaves:
            raise ValueError("both children must be leaves before merging")
        return IntervalPartition(tuple(sorted(
            [x for x in self.leaves if x not in (left, right)] + [parent]
        )))

    def coverage(self, segment_lengths_m: np.ndarray) -> PartitionCoverage:
        lengths = np.asarray(segment_lengths_m, dtype=np.float64)
        if lengths.shape != (5,) or np.any(~np.isfinite(lengths)) or np.any(lengths <= 0):
            return PartitionCoverage(False, False, ("invalid_segment_lengths",), len(self.leaves))
        reasons = []
        well_formed = True
        if len(set(self.leaves)) != len(self.leaves):
            reasons.append("duplicate_interval_id")
            well_formed = False
        for i, length in enumerate(lengths):
            current = sorted(
                (x.local_bounds(float(length)) for x in self.leaves if x.segment_id == i),
                key=lambda pair: pair[0],
            )
            if not current:
                reasons.append(f"missing_segment:{i}")
                continue
            if abs(current[0][0]) > 1e-12 or abs(current[-1][1] - length) > 1e-12:
                reasons.append(f"endpoint_gap:{i}")
            for (_, previous_end), (next_start, _) in zip(current, current[1:]):
                if next_start < previous_end - 1e-12:
                    reasons.append(f"overlap:{i}")
                    well_formed = False
                elif next_start > previous_end + 1e-12:
                    reasons.append(f"gap:{i}")
        return PartitionCoverage(well_formed, not bool(reasons),
                                 tuple(reasons), len(self.leaves))


@dataclass(frozen=True)
class IntervalSafetyRow:
    interval_id: str
    segment_id: int
    local_start_m: float
    local_end_m: float
    midpoint_arclength_m: float
    centerline_point_world: np.ndarray
    obb_witness_world: np.ndarray
    normal_box_to_point: np.ndarray
    point_signed_distance_m: float
    tube_radius_m: float
    coverage_term_m: float
    numerical_pad_m: float
    safe_distance_m: float
    distance_lower_bound_m: float
    h_m: float
    derivative_status: str
    shape_gradient: np.ndarray | None
    generalized_gradient: np.ndarray | None
    target_drift_m_s: float | None

    def to_dict(self) -> dict:
        return {
            "interval_id": self.interval_id,
            "segment_id": self.segment_id,
            "local_start_m": self.local_start_m,
            "local_end_m": self.local_end_m,
            "midpoint_arclength_m": self.midpoint_arclength_m,
            "centerline_point_world": self.centerline_point_world.tolist(),
            "obb_witness_world": self.obb_witness_world.tolist(),
            "normal_box_to_point": self.normal_box_to_point.tolist(),
            "point_signed_distance_m": self.point_signed_distance_m,
            "tube_radius_m": self.tube_radius_m,
            "coverage_term_m": self.coverage_term_m,
            "numerical_pad_m": self.numerical_pad_m,
            "safe_distance_m": self.safe_distance_m,
            "distance_lower_bound_m": self.distance_lower_bound_m,
            "h_m": self.h_m,
            "derivative_status": self.derivative_status,
            "shape_gradient": None if self.shape_gradient is None else self.shape_gradient.tolist(),
            "generalized_gradient": None if self.generalized_gradient is None else self.generalized_gradient.tolist(),
            "target_drift_m_s": self.target_drift_m_s,
        }


@dataclass(frozen=True)
class IntervalSafetyBatch:
    rows: tuple[IntervalSafetyRow, ...]
    interval_well_formed: bool
    coverage_complete: bool
    coverage_failure_reasons: tuple[str, ...]
    analytic_bound_assumptions_satisfied: bool
    floating_point_certification: str
    geometry_domain_status: str
    envelope_evidence_status: str
    subspace_residual_linf_rad: float
    point_evaluation_count: int
    shape_jacobian_evaluation_count: int
    mujoco_point_jacobian_count: int
    jacobian_evaluation_count: int
    total_query_time_ms: float

    @property
    def derivatives_supported(self) -> bool:
        return all(row.derivative_status == "SUPPORTED" for row in self.rows)

    @property
    def all_intervals_safe(self) -> bool:
        return (self.coverage_complete and self.analytic_bound_assumptions_satisfied
                and all(row.h_m >= 0.0 for row in self.rows))


def _obb_derivative_status(point_world: np.ndarray, box: OrientedBox) -> str:
    local = box.rotation.T @ (point_world - box.center)
    excess = np.abs(local) - box.half_extents
    if np.any(excess > 1e-14):
        return "SUPPORTED"  # Exterior projection onto a convex OBB is unique.
    margins = box.half_extents - np.abs(local)
    ordered = np.argsort(margins)
    if (margins[ordered[1]] - margins[ordered[0]] <= FEATURE_TIE_TOL_M
            or abs(local[ordered[0]]) <= FEATURE_TIE_TOL_M):
        return "NONSMOOTH_OBB_INTERIOR_FEATURE"
    return "SUPPORTED"


class FixedIntervalCBFEvaluator:
    """Evaluate the same fixed h for distance, 10D/17D derivative and drift."""

    def __init__(self, spec: RobotModelSpecV5, model: mujoco.MjModel) -> None:
        self.spec, self.model = spec, model
        self.shape_spec = default_continuum_model_spec(spec)
        self.shape_model = ContinuumShapeModel(self.shape_spec)
        self.qpos_ids, self.dof_ids = joint_addresses(model, spec)
        self.base_qpos_slice, self.base_dof_slice = free_joint_slices(model, spec.base_joint_name)
        _target_qpos, self.target_dof_slice = free_joint_slices(model, spec.target_free_joint_name)
        self.target_geom_id = int(mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "target_satellite_collision"
        ))
        if self.target_geom_id < 0:
            raise ValueError("target satellite OBB is missing")
        self.target_body_id = int(model.geom_bodyid[self.target_geom_id])
        self.base_body_id = int(mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, self.shape_spec.base_body_name
        ))
        if self.base_body_id < 0:
            raise ValueError("continuum floating base body is missing")

    def reaction_map(self, data: mujoco.MjData) -> np.ndarray:
        full_mass = np.zeros((self.model.nv, self.model.nv), dtype=np.float64)
        mujoco.mj_fullM(self.model, full_mass, data.qM)
        base_ids = np.arange(self.base_dof_slice.start, self.base_dof_slice.stop)
        arm_map = self.spec.planner_to_low_level
        base_map = -np.linalg.solve(full_mass[np.ix_(base_ids, base_ids)],
                                    full_mass[np.ix_(base_ids, self.dof_ids)] @ arm_map)
        result = np.zeros((self.model.nv, 17), dtype=np.float64)
        result[base_ids] = base_map
        result[self.dof_ids] = arm_map
        return result

    def _point_jacobian(self, data: mujoco.MjData, body_id: int,
                        point_world: np.ndarray) -> np.ndarray:
        result = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jac(self.model, data, result, None, point_world, body_id)
        return result

    def evaluate_state(self, data: mujoco.MjData, partition: IntervalPartition,
                       *, generalized_map: np.ndarray | None = None,
                       safe_distance_m: float = 0.005,
                       numerical_pad_m: float = NUMERICAL_PAD_M,
                       with_derivatives: bool = True,
                       derivative_interval_ids: set[str] | None = None) -> IntervalSafetyBatch:
        started = time.perf_counter()
        if not math.isfinite(safe_distance_m) or safe_distance_m <= 0:
            raise ValueError("safe distance must be positive and finite")
        if not math.isfinite(numerical_pad_m) or numerical_pad_m < 0:
            raise ValueError("numerical pad must be nonnegative and finite")
        coverage = partition.coverage(self.shape_spec.segment_lengths_m)
        low_level = np.asarray(data.qpos[self.qpos_ids[:CONTINUUM_ACTUATED_DOF]],
                               dtype=np.float64)
        projection = self.shape_spec.project_actual_configuration(low_level)
        q = projection.planner_configuration
        base_transform = transform_from_free_qpos(data.qpos[self.base_qpos_slice])
        box = target_box_from_mujoco(self.model, data, self.target_geom_id)
        in_domain = bool(np.all(q >= self.shape_spec.work_domain_lower_rad)
                         and np.all(q <= self.shape_spec.work_domain_upper_rad))
        in_subspace = projection.residual_linf_rad <= SHAPE_SUBSPACE_MEMBERSHIP_TOL_RAD
        domain = ("INSIDE_DECLARED_WORK_DOMAIN" if in_domain
                  else "OUTSIDE_DECLARED_WORK_DOMAIN")
        domain += (";ON_DECLARED_SHAPE_SUBSPACE" if in_subspace
                   else ";OUTSIDE_DECLARED_SHAPE_SUBSPACE")
        envelope = ("FINITE_V6_1A_REGRESSION_ONLY" if in_domain and in_subspace
                    else "NO_ENVELOPE_EVIDENCE_FOR_THIS_STATE")
        if generalized_map is None and with_derivatives:
            generalized_map = self.reaction_map(data)
        if generalized_map is not None:
            generalized_map = np.asarray(generalized_map, dtype=np.float64)
            if generalized_map.shape != (self.model.nv, 17) or np.any(~np.isfinite(generalized_map)):
                raise ValueError("generalized reaction map must be finite (nv,17)")
        target_qvel = np.zeros(self.model.nv, dtype=np.float64)
        target_qvel[self.target_dof_slice] = data.qvel[self.target_dof_slice]
        if np.any(~np.isfinite(target_qvel)):
            raise ValueError("target generalized velocity is nonfinite")

        rows = []
        point_count = 0
        shape_jac_count = 0
        mujoco_jac_count = 0
        boundaries = self.shape_spec.segment_boundaries_m
        ordered_intervals = sorted(partition.leaves)
        known_ids = {item.interval_id for item in ordered_intervals}
        if derivative_interval_ids is not None and not derivative_interval_ids <= known_ids:
            raise ValueError("derivative request includes an interval absent from the partition")
        midpoint_by_id = {}
        for interval in ordered_intervals:
            length = float(self.shape_spec.segment_lengths_m[interval.segment_id])
            a, b = interval.local_bounds(length)
            midpoint_by_id[interval.interval_id] = float(
                boundaries[interval.segment_id] + (a + b) / 2.0
            )
        selected = [item for item in ordered_intervals if with_derivatives
                    and (derivative_interval_ids is None
                         or item.interval_id in derivative_interval_ids)]
        selected_ids = {item.interval_id for item in selected}
        unselected = [item for item in ordered_intervals
                      if item.interval_id not in selected_ids]
        cached = {}
        if unselected:
            sampled = self.shape_model.batch_query(
                q, base_transform,
                [midpoint_by_id[item.interval_id] for item in unselected],
                with_jacobians=False,
            )
            cached = {item.interval_id: point for item, point in zip(unselected, sampled)}
        if selected:
            sampled = self.shape_model.batch_query(
                q, base_transform,
                [midpoint_by_id[item.interval_id] for item in selected],
                with_jacobians=True,
            )
            cached.update({item.interval_id: point for item, point in zip(selected, sampled)})
        for interval in ordered_intervals:
            length = float(self.shape_spec.segment_lengths_m[interval.segment_id])
            a, b = interval.local_bounds(length)
            midpoint = midpoint_by_id[interval.interval_id]
            request_derivative = interval.interval_id in selected_ids
            evaluation = cached[interval.interval_id]
            point_count += 1
            if request_derivative:
                shape_jac_count += 1
            point = evaluation.position_world
            signed = point_obb_signed_distance(point, box)
            radius = float(V61A_PCC_TUBE_RADII_M[interval.segment_id])
            coverage_term = (b - a) / 2.0  # K=1 from this PCC section model.
            lower = float(signed.signed_distance_m - radius - coverage_term - numerical_pad_m)
            h = lower - safe_distance_m
            derivative_status = _obb_derivative_status(point, box)
            shape_gradient = None
            generalized_gradient = None
            drift = None
            if not all(math.isfinite(x) for x in (lower, h)):
                derivative_status = "INVALID_NUMERIC"
            elif request_derivative and derivative_status == "SUPPORTED":
                normal = signed.normal_box_to_point
                shape_gradient = np.asarray(normal @ evaluation.position_jacobian,
                                            dtype=np.float64)
                base_jacobian = self._point_jacobian(data, self.base_body_id, point)
                generalized_gradient = np.asarray(
                    normal @ base_jacobian @ generalized_map, dtype=np.float64
                )
                generalized_gradient[:10] += shape_gradient
                target_jacobian = self._point_jacobian(
                    data, self.target_body_id, signed.box_point
                )
                mujoco_jac_count += 2
                drift = float(-normal @ target_jacobian @ target_qvel)
                if (np.any(~np.isfinite(shape_gradient))
                        or np.any(~np.isfinite(generalized_gradient))
                        or not math.isfinite(drift)):
                    derivative_status = "INVALID_NUMERIC"
                    shape_gradient = generalized_gradient = drift = None
            elif not request_derivative and derivative_status == "SUPPORTED":
                derivative_status = "NOT_REQUESTED"
            rows.append(IntervalSafetyRow(
                interval.interval_id, interval.segment_id, a, b, midpoint,
                point.copy(), signed.box_point.copy(), signed.normal_box_to_point.copy(),
                float(signed.signed_distance_m), radius, coverage_term,
                numerical_pad_m, safe_distance_m, lower, h, derivative_status,
                shape_gradient, generalized_gradient, drift,
            ))
        return IntervalSafetyBatch(
            rows=tuple(rows),
            interval_well_formed=coverage.interval_well_formed,
            coverage_complete=coverage.coverage_complete,
            coverage_failure_reasons=coverage.failure_reasons,
            analytic_bound_assumptions_satisfied=True,
            floating_point_certification="NOT_FORMALLY_CERTIFIED",
            geometry_domain_status=domain,
            envelope_evidence_status=envelope,
            subspace_residual_linf_rad=projection.residual_linf_rad,
            point_evaluation_count=point_count,
            shape_jacobian_evaluation_count=shape_jac_count,
            mujoco_point_jacobian_count=mujoco_jac_count,
            jacobian_evaluation_count=shape_jac_count + mujoco_jac_count,
            total_query_time_ms=(time.perf_counter() - started) * 1000.0,
        )
